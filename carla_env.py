import logging

import gymnasium as gym
import carla
import numpy as np
from gymnasium import spaces
import random
from typing import Dict, Tuple, Optional

from sensors import SensorManager, SensorPreprocessor
from reward import RewardCalculator


class CarlaEnv(gym.Env):
    """自定义 CARLA 环境，集成传感器管理和奖励计算"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 设置日志记录器
        self.logger = logging.getLogger(__name__)

        # 连接到 CARLA 服务器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # 如果当前地图不是指定的地图，则加载指定地图
        if self.world.get_map().name != config.env.sim_params['town']:
            self.world = self.client.load_world(config.env.sim_params['town'])

        # 设置仿真参数
        settings = self.world.get_settings()
        settings.synchronous_mode = config.env.sim_params['sync_mode']
        settings.fixed_delta_seconds = config.env.sim_params['fixed_delta_seconds']
        settings.no_rendering_mode = config.env.sim_params['no_rendering_mode']
        self.world.apply_settings(settings)

        # 获取蓝图
        self.blueprint_library = self.world.get_blueprint_library()

        # 设置动作空间
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),  # 方向盘、油门、刹车
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 设置观察空间
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(
                low=0,
                high=255,
                shape=(config.sensor.rgb_camera['height'],
                       config.sensor.rgb_camera['width'],
                       3),
                dtype=np.uint8
            ),
            'lidar': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(config.model.lidar_params['in_points'],
                       config.model.lidar_params['point_dim']),
                dtype=np.float32
            )
        })

        # 初始化组件
        self.vehicle = None
        self.sensor_manager = None
        self.sensor_preprocessor = SensorPreprocessor(config.sensor)
        self.reward_calculator = RewardCalculator(config)

        # 记录当前回合信息
        self.episode_step = 0
        self.max_steps = config.training.base_params['n_steps_per_episode']

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)

        # 清理现有的传感器和车辆
        if hasattr(self, 'sensor_manager') and self.sensor_manager is not None:
            self.sensor_manager.destroy()
            self.sensor_manager = None

        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None

        # 等待一帧以确保清理完成
        self.world.tick()

        # 生成车辆
        try:
            vehicle_bp = self.blueprint_library.find(
                self.config.env.vehicle_params['model']
            )
            spawn_points = self.world.get_map().get_spawn_points()

            if self.config.env.vehicle_params['spawn_point_index'] is not None:
                spawn_point = spawn_points[self.config.env.vehicle_params['spawn_point_index']]
            else:
                spawn_point = random.choice(spawn_points)

            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

            # 等待车辆生成
            self.world.tick()

            # 初始化传感器管理器
            self.sensor_manager = SensorManager(
                self.world,
                self.vehicle,
                self.config.sensor
            )

            # 等待传感器初始化
            self.world.tick()

            # 获取初始观察
            try:
                obs = self.sensor_manager.get_data(timeout=2.0)
            except TimeoutError:
                # 如果获取数据超时，重试重置
                return self.reset()

            # 预处理传感器数据
            processed_obs = {
                'rgb': self.sensor_preprocessor.process_rgb(obs['rgb']),
                'lidar': self.sensor_preprocessor.process_lidar(obs['lidar'])
            }

            # 重置回合计数器和奖励计算器
            self.episode_step = 0
            self.reward_calculator.reset()

            return processed_obs, {}

        except Exception as e:
            # 如果生成过程中出现错误，清理并重试
            if hasattr(self, 'sensor_manager') and self.sensor_manager is not None:
                self.sensor_manager.destroy()
                self.sensor_manager = None

            if hasattr(self, 'vehicle') and self.vehicle is not None:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
                self.vehicle = None

            self.world.tick()
            return self.reset()

    def step(self, action):
        """执行动作并返回结果"""
        try:
            self.episode_step += 1

            # 应用车辆控制
            control = carla.VehicleControl(
                throttle=float(action[1]),
                steer=float(action[0]),
                brake=float(action[2]),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(control)

            # 等待物理模拟更新
            self.world.tick()

            # 等待传感器数据
            try:
                obs = self.sensor_manager.get_data(timeout=2.0)
            except TimeoutError:
                # 如果获取数据超时，认为是episode结束
                return self.reset()[0], 0.0, True, True, {}

            # 预处理传感器数据
            processed_obs = {
                'rgb': self.sensor_preprocessor.process_rgb(obs['rgb']),
                'lidar': self.sensor_preprocessor.process_lidar(obs['lidar'])
            }

            # 获取车辆状态
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_velocity = self.vehicle.get_velocity()

            # 获取导航信息
            waypoint = self.world.get_map().get_waypoint(vehicle_location)

            # 检测碰撞和车道入侵
            collision_detected = self._check_collision()
            lane_invasion = self._check_lane_invasion()

            # 获取周围车辆
            surrounding_vehicles = self._get_surrounding_vehicles()

            # 计算奖励
            reward, reward_info = self.reward_calculator.calculate_reward(
                self.vehicle,
                waypoint,
                collision_detected,
                lane_invasion,
                surrounding_vehicles
            )

            # 判断是否结束
            terminated = collision_detected or self.episode_step >= self.max_steps
            truncated = False  # 用于提前终止的情况

            # 收集额外信息
            info = {
                'reward_info': reward_info,
                'collision': collision_detected,
                'lane_invasion': lane_invasion,
                'step': self.episode_step,
                'vehicle_location': [vehicle_location.x, vehicle_location.y, vehicle_location.z],
                'vehicle_velocity': [vehicle_velocity.x, vehicle_velocity.y, vehicle_velocity.z]
            }

            return processed_obs, reward, terminated, truncated, info

        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}")
            # 发生错误时重置环境
            return self.reset()[0], 0.0, True, True, {'error': str(e)}

    def _check_collision(self):
        """检查是否发生碰撞"""
        # 实现碰撞检测逻辑
        return False

    def _check_lane_invasion(self):
        """检查是否压线或越线"""
        # 实现车道线检测逻辑
        return False

    def _get_surrounding_vehicles(self):
        """获取周围的车辆"""
        return []

    def close(self):
        """清理环境"""
        if hasattr(self, 'sensor_manager') and self.sensor_manager is not None:
            self.sensor_manager.destroy()
            self.sensor_manager = None

        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None

        # 恢复世界设置
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        # 等待一帧以确保清理完成
        self.world.tick()
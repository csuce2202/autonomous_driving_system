import numpy as np
import carla
from typing import Dict, Tuple, Optional
import math


class RewardCalculator:
    """奖励计算器：整合多个子奖励函数"""

    def __init__(self, config):
        self.config = config

        # 初始化状态变量
        self.last_location = None
        self.last_velocity = None
        self.last_transform = None
        self.cumulative_lane_invasion = 0
        self.collision_history = []

        # 从配置加载参数或使用默认值
        if hasattr(config, 'env') and hasattr(config.env, 'vehicle_params'):
            self.target_speed = config.env.vehicle_params.get('target_speed', 30.0)
            self.min_ttc = config.env.vehicle_params.get('min_ttc', 2.0)
            self.max_acceleration = config.env.vehicle_params.get('max_acceleration', 3.0)
            self.max_steering_angle = config.env.vehicle_params.get('max_steering_angle', 45.0)
        else:
            self.target_speed = 30.0  # km/h
            self.min_ttc = 2.0  # seconds
            self.max_acceleration = 3.0  # m/s²
            self.max_steering_angle = 45.0  # degrees

        # 设置奖励权重
        if hasattr(config, 'training') and hasattr(config.training, 'reward_weights'):
            self.weights = config.training.reward_weights
        else:
            self.weights = {
                'distance': 1.0,
                'speed': 0.5,
                'collision': 10.0,
                'lane': 2.0,
                'comfort': 0.3,
                'ttc': 1.0
            }

    def reset(self):
        """重置状态"""
        self.last_location = None
        self.last_velocity = None
        self.last_transform = None
        self.cumulative_lane_invasion = 0
        self.collision_history.clear()

    def calculate_reward(self,
                         vehicle: carla.Vehicle,
                         waypoint: carla.Waypoint,
                         collision_detected: bool,
                         lane_invasion: bool,
                         surrounding_vehicles: list) -> Tuple[float, Dict[str, float]]:
        """
        计算总体奖励

        Args:
            vehicle: CARLA车辆对象
            waypoint: 当前目标路点
            collision_detected: 是否检测到碰撞
            lane_invasion: 是否压线或越线
            surrounding_vehicles: 周围车辆列表

        Returns:
            总奖励值和各分量奖励的字典
        """
        current_location = vehicle.get_location()
        current_velocity = vehicle.get_velocity()
        current_transform = vehicle.get_transform()

        # 计算各个奖励分量
        reward_components = {}

        # 1. 导航距离奖励
        reward_components['distance'] = self._calculate_distance_reward(
            current_location,
            waypoint.transform.location
        )

        # 2. 速度奖励
        reward_components['speed'] = self._calculate_speed_reward(current_velocity)

        # 3. 碰撞惩罚
        reward_components['collision'] = self._calculate_collision_penalty(
            collision_detected,
            current_location
        )

        # 4. 车道保持奖励
        reward_components['lane'] = self._calculate_lane_keeping_reward(
            current_location,
            waypoint,
            lane_invasion
        )

        # 5. 舒适性奖励
        reward_components['comfort'] = self._calculate_comfort_reward(
            current_velocity,
            current_transform
        )

        # 6. TTC奖励
        reward_components['ttc'] = self._calculate_ttc_reward(
            vehicle,
            surrounding_vehicles
        )

        # 更新状态
        self.last_location = current_location
        self.last_velocity = current_velocity
        self.last_transform = current_transform

        if collision_detected:
            self.collision_history.append(current_location)

        if lane_invasion:
            self.cumulative_lane_invasion += 1

        # 计算加权总奖励
        total_reward = sum(
            self.weights[key] * value
            for key, value in reward_components.items()
        )

        return total_reward, reward_components

    def _calculate_distance_reward(self,
                                   current_location: carla.Location,
                                   target_location: carla.Location) -> float:
        """计算导航距离奖励"""
        distance = current_location.distance(target_location)

        # 使用平滑的奖励函数
        reward = math.exp(-distance / 10.0)

        # 如果有上一帧的位置，计算是否在接近目标
        if self.last_location is not None:
            prev_distance = self.last_location.distance(target_location)
            progress = prev_distance - distance
            reward += np.clip(progress, -1.0, 1.0)

        return reward

    def _calculate_speed_reward(self, velocity: carla.Vector3D) -> float:
        """计算速度奖励"""
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2)  # km/h

        # 速度偏差惩罚
        speed_diff = abs(speed - self.target_speed)
        speed_reward = -speed_diff / self.target_speed

        # 额外奖励保持稳定速度
        if self.last_velocity is not None:
            last_speed = 3.6 * math.sqrt(
                self.last_velocity.x ** 2 +
                self.last_velocity.y ** 2
            )
            speed_stability = -abs(speed - last_speed) / 5.0  # 惩罚突然的速度变化
            speed_reward += speed_stability

        return speed_reward

    def _calculate_collision_penalty(self,
                                     collision_detected: bool,
                                     current_location: carla.Location) -> float:
        """计算碰撞惩罚"""
        if not collision_detected:
            return 0.0

        # 检查是否是新的碰撞
        if len(self.collision_history) > 0:
            last_collision = self.collision_history[-1]
            if current_location.distance(last_collision) < 1.0:
                return -50.0  # 较轻的惩罚持续碰撞

        return -100.0  # 严重惩罚新的碰撞

    def _calculate_lane_keeping_reward(self,
                                       current_location: carla.Location,
                                       waypoint: carla.Waypoint,
                                       lane_invasion: bool) -> float:
        """计算车道保持奖励"""
        if lane_invasion:
            self.cumulative_lane_invasion += 1
            base_penalty = -1.0
        else:
            base_penalty = 0.0

        # 计算与车道中心线的横向距离
        vehicle_loc = np.array([current_location.x, current_location.y])
        waypoint_loc = np.array([waypoint.transform.location.x,
                                 waypoint.transform.location.y])

        lateral_distance = np.linalg.norm(vehicle_loc - waypoint_loc)

        # 使用高斯函数计算奖励
        distance_reward = math.exp(-lateral_distance ** 2 / 2.0)

        # 累积压线惩罚
        cumulative_penalty = -0.1 * self.cumulative_lane_invasion

        return base_penalty + distance_reward + cumulative_penalty

    def _calculate_comfort_reward(self,
                                  current_velocity: carla.Vector3D,
                                  current_transform: carla.Transform) -> float:
        """计算舒适性奖励"""
        if self.last_velocity is None or self.last_transform is None:
            return 0.0

        # 计算加速度
        dt = self.config.env.sim_params['fixed_delta_seconds']
        acceleration = (current_velocity - self.last_velocity) / dt
        acc_magnitude = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2)

        # 计算角速度（转向速率）
        yaw_diff = current_transform.rotation.yaw - self.last_transform.rotation.yaw
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360
        angular_velocity = abs(yaw_diff) / dt

        # 计算惩罚
        acc_penalty = -acc_magnitude / self.max_acceleration
        steering_penalty = -angular_velocity / self.max_steering_angle

        return acc_penalty + steering_penalty

    def _calculate_ttc_reward(self,
                              vehicle: carla.Vehicle,
                              surrounding_vehicles: list) -> float:
        """计算TTC（Time To Collision）奖励"""
        if not surrounding_vehicles:
            return 1.0

        min_ttc = float('inf')
        ego_velocity = vehicle.get_velocity()
        ego_location = vehicle.get_location()

        for other_vehicle in surrounding_vehicles:
            other_velocity = other_vehicle.get_velocity()
            other_location = other_vehicle.get_location()

            # 计算相对速度
            relative_velocity = math.sqrt(
                (ego_velocity.x - other_velocity.x) ** 2 +
                (ego_velocity.y - other_velocity.y) ** 2
            )

            # 计算距离
            distance = ego_location.distance(other_location)

            # 计算TTC
            if relative_velocity > 0:
                ttc = distance / relative_velocity
                min_ttc = min(min_ttc, ttc)

        # TTC奖励计算
        if min_ttc == float('inf'):
            return 1.0
        elif min_ttc < self.min_ttc:
            return -1.0 * (self.min_ttc - min_ttc) / self.min_ttc
        else:
            return math.exp(-(min_ttc - self.min_ttc))

    def get_reward_info(self) -> Dict[str, float]:
        """获取当前回合的奖励统计信息"""
        return {
            'cumulative_lane_invasions': self.cumulative_lane_invasion,
            'collision_count': len(self.collision_history)
        }
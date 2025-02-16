"""
File: carla_pettingzoo_env.py

Description:
  A PettingZoo ParallelEnv environment for multi-agent CARLA with TTC & PET.
  We rename `num_agents` -> `_num_agents` to avoid conflict with PettingZoo parent's property.
"""

import numpy as np
from pettingzoo.utils.env import ParallelEnv
try:
    import carla
except ImportError:
    raise RuntimeError("Carla Python API not found or not installed.")

import gymnasium as gym
import math

from reward_functions import compute_multi_agent_rewards


class CarlaPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "carla_pettingzoo_env"}

    def __init__(
        self,
        host="localhost",
        port=2000,
        agent_count=3,      # <-- 改名，不使用num_agents
        obs_dim=8,
        action_dim=3,
        max_steps=1000,
        noise_std=0.0,
        render_mode=None
    ):
        super().__init__()
        self.host = host
        self.port = port
        self._num_agents = agent_count  # <-- 用 _num_agents 存储实际数量
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.render_mode = render_mode

        # 建立 PettingZoo 所需的代理列表
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        # 定义 observation & action spaces
        single_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        single_act_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.observation_spaces = {agent: single_obs_space for agent in self.agents}
        self.action_spaces = {agent: single_act_space for agent in self.agents}

        # Carla 相关
        self.client = None
        self.world = None
        self.vehicles_list = []

        # 碰撞标记
        self.collisions = [False] * self._num_agents
        self.step_count = 0

        # 终止 & 截断 标记
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self._init_carla()

    def _init_carla(self):
        print("[INFO] Connecting to CARLA for PettingZoo multi-agent environment...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        # 移除旧车辆
        for actor in self.world.get_actors().filter("vehicle.*"):
            actor.destroy()

        # 生成车辆
        spawn_points = self.world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        self.vehicles_list = []

        for i in range(self._num_agents):
            transform = spawn_points[i % len(spawn_points)]
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if vehicle is None:
                raise RuntimeError(f"Unable to spawn vehicle {i}. Possibly insufficient spawn points.")
            self.vehicles_list.append(vehicle)
            self._attach_collision_sensor(vehicle, i)

        print(f"[INFO] Spawned {len(self.vehicles_list)} vehicles for {self._num_agents} agents.")

    def _attach_collision_sensor(self, vehicle, idx):
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

        def on_collision(event):
            self.collisions[idx] = True

        collision_sensor.listen(on_collision)

    def render(self):
        if self.render_mode == "human":
            pass

    def reset(self, seed=None, options=None):
        self._reset_env(seed)
        return self._get_obs_dict()

    def _reset_env(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.collisions = [False] * self._num_agents
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # 重新放置车辆
        spawn_points = self.world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)
        for i, vehicle in enumerate(self.vehicles_list):
            vehicle.set_transform(spawn_points[i % len(spawn_points)])
            # 使用 set_target_velocity 重置速度
            try:
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            except AttributeError:
                print("[WARNING] set_target_velocity not available, skipping velocity reset.")
            # 如果有对应接口，可以重置角速度；否则可以选择移除此行
            try:
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            except AttributeError:
                print("[WARNING] set_target_angular_velocity not available, skipping angular velocity reset.")

        self.world.tick()

    def step(self, actions):
        self.step_count += 1

        # 1) apply actions
        for i, agent in enumerate(self.agents):
            if not self.terminations[agent] and not self.truncations[agent]:
                action = actions[agent]
                self._apply_action(self.vehicles_list[i], action)

        self.world.tick()
        obs = self._get_obs_dict()

        # 2) compute reward
        collisions_list = [self.collisions[i] for i in range(self._num_agents)]
        speeds_list = [self._compute_speed(self.vehicles_list[i]) for i in range(self._num_agents)]
        lane_dev_list = [self._compute_lane_deviation(self.vehicles_list[i]) for i in range(self._num_agents)]
        ttc_list = [self._compute_ttc_for_agent(i) for i in range(self._num_agents)]
        pet_list = [self._compute_pet_for_agent(i) for i in range(self._num_agents)]

        rewards_dict = compute_multi_agent_rewards(
            collisions=collisions_list,
            speeds=speeds_list,
            lane_deviations=lane_dev_list,
            ttcs=ttc_list,
            pets=pet_list,
            desired_speed=10.0,
            collision_penalty=-10.0,
            speed_factor=0.1,
            lane_dev_factor=-0.05,
            ttc_threshold=3.0,
            ttc_factor=-1.0,
            pet_threshold=2.0,
            pet_factor=-0.5
        )

        # 3) update done/trunc
        for i, agent in enumerate(self.agents):
            if self.collisions[i]:
                self.terminations[agent] = True
            if self.step_count >= self.max_steps:
                self.truncations[agent] = True

        done_all = all(self.terminations[a] or self.truncations[a] for a in self.agents)

        # 4) build info
        infos = {}
        for i, agent in enumerate(self.agents):
            infos[agent] = {
                "collision": self.collisions[i],
                "step": self.step_count,
                "ttc": ttc_list[i],
                "pet": pet_list[i],
                "lane_dev": lane_dev_list[i],
            }

        return obs, rewards_dict, self.terminations, self.truncations, infos

    def _apply_action(self, vehicle, action):
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        if self.noise_std > 0.0:
            throttle += np.random.randn() * self.noise_std
            steer += np.random.randn() * self.noise_std
            brake += np.random.randn() * self.noise_std

        throttle = float(np.clip(throttle, 0.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        vehicle.apply_control(control)

    def _get_obs_dict(self):
        obs_dict = {}
        for i, agent_id in enumerate(self.agents):
            obs_dict[agent_id] = self._get_single_agent_obs(self.vehicles_list[i])
        return obs_dict

    def _get_single_agent_obs(self, vehicle):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        obs_vec = np.array([x, y, yaw, speed, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return obs_vec

    def _compute_speed(self, vehicle):
        vel = vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def _compute_lane_deviation(self, vehicle):
        loc = vehicle.get_transform().location
        waypoint = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return 0.0
        lane_center = waypoint.transform.location
        dx = loc.x - lane_center.x
        dy = loc.y - lane_center.y
        return math.sqrt(dx*dx + dy*dy)

    def _compute_ttc_for_agent(self, i):
        ego_vehicle = self.vehicles_list[i]
        ego_loc = ego_vehicle.get_transform().location
        ego_vel = ego_vehicle.get_velocity()
        ego_xy = np.array([ego_loc.x, ego_loc.y])
        ego_vxy = np.array([ego_vel.x, ego_vel.y])

        min_ttc = 999.0
        for j in range(self._num_agents):
            if j == i:
                continue
            other = self.vehicles_list[j]
            other_loc = other.get_transform().location
            other_vel = other.get_velocity()
            other_xy = np.array([other_loc.x, other_loc.y])
            other_vxy = np.array([other_vel.x, other_vel.y])

            rel_pos = other_xy - ego_xy
            rel_vel = other_vxy - ego_vxy
            dist = np.linalg.norm(rel_pos)
            speed_rel = np.linalg.norm(rel_vel)

            if speed_rel > 0.1:
                ttc = dist / speed_rel
                if ttc < min_ttc:
                    min_ttc = ttc

        return float(min_ttc)

    def _compute_pet_for_agent(self, i):
        ego_vehicle = self.vehicles_list[i]
        ego_loc = ego_vehicle.get_transform().location
        for j in range(self._num_agents):
            if j == i:
                continue
            other = self.vehicles_list[j]
            other_loc = other.get_transform().location
            dist = math.sqrt((ego_loc.x - other_loc.x)**2 + (ego_loc.y - other_loc.y)**2)
            if dist < 2.0:
                return 0.5
        return 999.0

    def close(self):
        print("[INFO] Closing CarlaPettingZooEnv.")
        for v in self.vehicles_list:
            if v.is_alive:
                v.destroy()
        self.vehicles_list = []
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        print("[INFO] Environment closed.")

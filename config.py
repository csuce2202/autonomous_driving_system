from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np


@dataclass
class SensorConfig:
    """传感器配置"""

    def __init__(self):
        # RGB相机配置
        self.rgb_camera = {
            'width': 84,
            'height': 84,
            'fov': 90,
            'location': (1.5, 0.0, 2.4),  # x, y, z
            'rotation': (0.0, 0.0, 0.0),  # pitch, yaw, roll
            'fps': 20
        }

        # LiDAR配置
        self.lidar = {
            'channels': 32,
            'range': 20.0,  # meters
            'points_per_second': 100000,
            'rotation_frequency': 20,
            'upper_fov': 10.0,
            'lower_fov': -30.0,
            'location': (1.5, 0.0, 2.4),
            'rotation': (0.0, 0.0, 0.0)
        }


@dataclass
class EnvConfig:
    """环境配置"""
    # 仿真配置
    sim_params = {
        'town': 'Town10HD_Opt',
        'fps': 20,
        'sync_mode': True,
        'no_rendering_mode': False,
        'fixed_delta_seconds': 0.05  # 1/fps
    }

    # 车辆配置
    vehicle_params = {
        'model': 'vehicle.tesla.model3',
        'role_name': 'hero',
        'spawn_point_index': None  # None表示随机
    }

    # 天气配置
    weather_params = {
        'cloudiness': 0,
        'precipitation': 0,
        'precipitation_deposits': 0,
        'wind_intensity': 0,
        'sun_azimuth_angle': 45,
        'sun_altitude_angle': 45
    }


@dataclass
class TrainingConfig:
    """训练配置"""

    def __init__(self):
        # 基础训练参数
        self.base_params = {
            'total_timesteps': 1_000_000,
            'n_steps_per_episode': 1000,
            'eval_interval': 10000,
            'save_interval': 50000,
            'log_interval': 1000
        }

        # PPO算法参数
        self.ppo_params = {
            'n_epochs': 10,
            'batch_size': 64,
            'learning_rate': 3e-4,
            'clip_range': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'gamma': 0.99,
            'gae_lambda': 0.95
        }

        # 奖励函数权重
        self.reward_weights = {
            'distance': 1.0,
            'speed': 0.5,
            'collision': 10.0,
            'lane': 2.0,
            'comfort': 0.3,
            'ttc': 1.0
        }


@dataclass
class ModelConfig:
    """模型配置"""
    # CNN特征提取器参数
    cnn_params = {
        'in_channels': 3,
        'base_channels': 32,
        'n_conv_layers': 4,
        'kernel_sizes': [4, 4, 4, 3],
        'strides': [2, 2, 2, 2],
        'paddings': [1, 1, 1, 1]
    }

    # LiDAR特征提取器参数
    lidar_params = {
        'in_points': 100,
        'point_dim': 3,
        'hidden_dims': [64, 128, 256, 512]
    }

    # 策略网络参数
    policy_params = {
        'hidden_dims': [512, 256, 128],
        'action_dim': 3
    }


@dataclass
class LogConfig:
    """日志配置"""
    # 日志参数
    log_params = {
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'tensorboard_dir': 'runs',
        'eval_video_dir': 'eval_videos',
        'log_level': 'INFO'
    }

    # 评估配置
    eval_params = {
        'n_eval_episodes': 5,
        'record_video': True,
        'video_fps': 20
    }


class Config:
    """总配置类"""
    def __init__(self):
        self.sensor = SensorConfig()
        self.env = EnvConfig()
        self.training = TrainingConfig()
        self.model = ModelConfig()
        self.log = LogConfig()

    @property
    def device(self):
        """返回计算设备"""
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
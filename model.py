import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器：处理视觉输入"""

    def __init__(self, config):
        super().__init__()

        cnn_params = config.model.cnn_params
        layers = []

        # 动态构建CNN层
        in_channels = cnn_params['in_channels']
        current_channels = cnn_params['base_channels']

        for i in range(cnn_params['n_conv_layers']):
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=current_channels,
                    kernel_size=cnn_params['kernel_sizes'][i],
                    stride=cnn_params['strides'][i],
                    padding=cnn_params['paddings'][i]
                ),
                nn.ReLU()
            ])
            in_channels = current_channels
            current_channels *= 2

        self.cnn = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # 计算输出维度
        with torch.no_grad():
            test_input = torch.zeros(1, cnn_params['in_channels'],
                                     config.sensor.rgb_camera['height'],
                                     config.sensor.rgb_camera['width'])
            test_output = self.flatten(self.cnn(test_input))
            self.output_dim = test_output.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        return self.flatten(x)


class LiDARFeatureExtractor(nn.Module):
    """LiDAR特征提取器"""

    def __init__(self, config):
        super().__init__()

        lidar_params = config.model.lidar_params
        self.input_transform = nn.Sequential(
            nn.Linear(lidar_params['point_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        layers = []
        prev_dim = 64
        for hidden_dim in lidar_params['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.feature_extract = nn.Sequential(*layers)
        self.output_dim = lidar_params['hidden_dims'][-1]

    def forward(self, x):
        x = self.input_transform(x)
        features = self.feature_extract(x)
        return torch.max(features, dim=1)[0]  # max pooling across points


class PolicyNetwork(nn.Module):
    """策略网络"""

    def __init__(self, config):
        super().__init__()

        policy_params = config.model.policy_params
        prev_dim = policy_params['hidden_dims'][0]

        layers = []
        for hidden_dim in policy_params['hidden_dims'][1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # 动作头：均值和标准差
        self.action_mean = nn.Linear(prev_dim, policy_params['action_dim'])
        self.action_std = nn.Sequential(
            nn.Linear(prev_dim, policy_params['action_dim']),
            nn.Softplus()
        )

    def forward(self, features):
        x = self.shared(features)
        return self.action_mean(x), self.action_std(x)


class ValueNetwork(nn.Module):
    """价值网络"""

    def __init__(self, config):
        super().__init__()

        policy_params = config.model.policy_params
        prev_dim = policy_params['hidden_dims'][0]

        layers = []
        for hidden_dim in policy_params['hidden_dims'][1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*layers)

    def forward(self, features):
        return self.value_net(features)


class AutoDrivingNetwork(nn.Module):
    """自动驾驶网络：结合所有组件"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 特征提取器
        self.cnn_extractor = CNNFeatureExtractor(config)
        self.lidar_extractor = LiDARFeatureExtractor(config)

        # 特征融合
        combined_dim = self.cnn_extractor.output_dim + self.lidar_extractor.output_dim
        self.config.model.policy_params['hidden_dims'][0] = combined_dim

        # 策略和价值网络
        self.policy_net = PolicyNetwork(config)
        self.value_net = ValueNetwork(config)

        # 将模型移动到指定设备
        self.device = config.device
        self.to(self.device)

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            observations: 包含'rgb'和'lidar'数据的字典

        Returns:
            action_mean: 动作分布均值
            action_std: 动作分布标准差
            value: 状态值
        """
        # 特征提取
        rgb_features = self.cnn_extractor(observations['rgb'])
        lidar_features = self.lidar_extractor(observations['lidar'])

        # 特征融合
        combined_features = torch.cat([rgb_features, lidar_features], dim=-1)

        # 获取策略分布参数和状态值
        action_mean, action_std = self.policy_net(combined_features)
        value = self.value_net(combined_features)

        return action_mean, action_std, value

    def get_action(self, observations: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[
        torch.Tensor, Dict]:
        """
        根据观察选择动作

        Args:
            observations: 观察数据
            deterministic: 是否使用确定性策略

        Returns:
            action: 选择的动作
            action_info: 包含额外信息的字典
        """
        action_mean, action_std, value = self(observations)

        if deterministic:
            action = torch.tanh(action_mean)
        else:
            dist = Normal(action_mean, action_std)
            action = torch.tanh(dist.sample())

        action_info = {
            'mean': action_mean,
            'std': action_std,
            'value': value,
            'log_prob': dist.log_prob(action) if not deterministic else None
        }

        return action, action_info

    def evaluate_actions(self,
                         observations: Dict[str, torch.Tensor],
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作

        Args:
            observations: 观察数据
            actions: 要评估的动作

        Returns:
            log_prob: 动作的对数概率
            entropy: 策略的熵
            value: 状态值
        """
        action_mean, action_std, value = self(observations)
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value


class PPOMemory:
    """PPO算法的经验回放缓冲区"""

    def __init__(self, config):
        self.config = config
        self.clear()

    def add(self, observations, action, reward, value, log_prob, done):
        """添加一个转换"""
        self.observations['rgb'].append(observations['rgb'])
        self.observations['lidar'].append(observations['lidar'])
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """清空缓冲区"""
        self.observations = {'rgb': [], 'lidar': []}
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """获取批量数据"""
        batch = {
            'observations': {
                'rgb': torch.stack(self.observations['rgb']),
                'lidar': torch.stack(self.observations['lidar'])
            },
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards),
            'values': torch.stack(self.values),
            'log_probs': torch.stack(self.log_probs),
            'dones': torch.tensor(self.dones)
        }
        return batch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime


class Visualizer:
    def __init__(self, log_dir: str, config):
        """
        初始化可视化器

        Args:
            log_dir: 日志目录路径
            config: 配置对象
        """
        self.log_dir = Path(log_dir)
        self.config = config

        # 初始化数据存储
        self.episode_rewards = []
        self.episode_lengths = []
        self.timestamps = []
        self.eval_rewards = []
        self.eval_timestamps = []

        # 用于存储详细的奖励信息
        self.reward_components = {
            'distance': [],
            'speed': [],
            'collision': [],
            'lane': [],
            'comfort': [],
            'ttc': []
        }

        # 用于存储动作信息
        self.actions = {
            'steering': [],
            'throttle': [],
            'brake': []
        }

        # 创建图形和子图
        self.setup_plots()

    def setup_plots(self):
        """设置可视化图表"""
        # 创建主图形和子图
        plt.ion()  # 打开交互模式
        self.fig = plt.figure(figsize=(15, 10))

        # 创建子图网格
        gs = self.fig.add_gridspec(3, 2)

        # 训练奖励图表
        self.ax_reward = self.fig.add_subplot(gs[0, :])
        self.ax_reward.set_title('Training Progress')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')

        # 奖励分量图表
        self.ax_components = self.fig.add_subplot(gs[1, :])
        self.ax_components.set_title('Reward Components')
        self.ax_components.set_xlabel('Episode')
        self.ax_components.set_ylabel('Component Value')

        # 动作图表
        self.ax_actions = self.fig.add_subplot(gs[2, :])
        self.ax_actions.set_title('Actions')
        self.ax_actions.set_xlabel('Episode')
        self.ax_actions.set_ylabel('Action Value')

        # 传感器数据可视化
        self.fig_sensors = plt.figure(figsize=(12, 5))

        # RGB相机图像
        self.ax_rgb = self.fig_sensors.add_subplot(121)
        self.ax_rgb.set_title('RGB Camera')

        # LiDAR点云
        self.ax_lidar = self.fig_sensors.add_subplot(122, projection='3d')
        self.ax_lidar.set_title('LiDAR Point Cloud')

        # 调整布局
        self.fig.tight_layout()
        self.fig_sensors.tight_layout()

        # 显示图形
        plt.show()

    def visualize_sensor_data(self, rgb_image: np.ndarray, lidar_points: np.ndarray):
        """
        可视化传感器数据

        Args:
            rgb_image: RGB图像数据，形状为(C, H, W)或(H, W, C)
            lidar_points: LiDAR点云数据
        """
        # 转换图像格式
        if rgb_image.ndim == 3:
            if rgb_image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                rgb_image = rgb_image.transpose(1, 2, 0)

            # 如果是归一化的数据，转回0-255范围
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)

        # 显示RGB图像
        self.ax_rgb.clear()
        self.ax_rgb.imshow(rgb_image)
        self.ax_rgb.axis('off')

        # 显示LiDAR点云
        self.ax_lidar.clear()
        if len(lidar_points) > 0:
            # 确保点云数据是正确的形状
            if lidar_points.ndim == 2 and lidar_points.shape[1] >= 3:
                scatter = self.ax_lidar.scatter(
                    lidar_points[:, 0],  # x坐标
                    lidar_points[:, 1],  # y坐标
                    lidar_points[:, 2],  # z坐标
                    c=lidar_points[:, 2],  # 使用高度作为颜色
                    cmap='viridis',
                    marker='.',
                    s=1  # 点的大小
                )

                # 设置视角和标签
                self.ax_lidar.view_init(elev=30, azim=45)
                self.ax_lidar.set_xlabel('X')
                self.ax_lidar.set_ylabel('Y')
                self.ax_lidar.set_zlabel('Z')

                # 设置轴的范围
                if hasattr(self.config.sensor.lidar, 'range'):
                    range_limit = self.config.sensor.lidar['range']
                    self.ax_lidar.set_xlim([-range_limit, range_limit])
                    self.ax_lidar.set_ylim([-range_limit, range_limit])
                    self.ax_lidar.set_zlim([-range_limit / 2, range_limit / 2])

        # 更新图表
        try:
            self.fig_sensors.canvas.draw()
            self.fig_sensors.canvas.flush_events()
        except Exception as e:
            logging.warning(f"Error updating sensor visualization: {e}")

    def update_training_info(self, episode: int, episode_reward: float,
                             reward_info: Dict[str, float], action: np.ndarray):
        """
        更新训练信息

        Args:
            episode: 当前episode数
            episode_reward: episode总奖励
            reward_info: 奖励分量信息
            action: 执行的动作
        """
        # 更新基础指标
        self.episode_rewards.append(episode_reward)
        self.timestamps.append(episode)

        # 更新奖励分量
        for component in self.reward_components.keys():
            if component in reward_info:
                self.reward_components[component].append(reward_info[component])

        # 更新动作信息
        if len(action) >= 3:
            self.actions['steering'].append(action[0])
            self.actions['throttle'].append(action[1])
            self.actions['brake'].append(action[2])

        # 更新图表
        if episode % self.config.log.log_params.get('plot_interval', 1) == 0:
            self._update_plots()

    def _update_plots(self):
        """更新所有图表"""
        # 清除所有子图
        self.ax_reward.clear()
        self.ax_components.clear()
        self.ax_actions.clear()

        # 绘制奖励曲线
        self.ax_reward.plot(self.timestamps, self.episode_rewards, 'b-', label='Episode Reward')
        if self.eval_rewards:
            self.ax_reward.plot(self.eval_timestamps, self.eval_rewards, 'r-', label='Eval Reward')
        self.ax_reward.set_title('Training Progress')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.legend()
        self.ax_reward.grid(True)

        # 绘制奖励分量
        for component, values in self.reward_components.items():
            if values:  # 只绘制有数据的分量
                self.ax_components.plot(self.timestamps[-len(values):], values,
                                        label=component)
        self.ax_components.set_title('Reward Components')
        self.ax_components.set_xlabel('Episode')
        self.ax_components.set_ylabel('Component Value')
        self.ax_components.legend()
        self.ax_components.grid(True)

        # 绘制动作
        for action_type, values in self.actions.items():
            if values:  # 只绘制有数据的动作
                self.ax_actions.plot(self.timestamps[-len(values):], values,
                                     label=action_type)
        self.ax_actions.set_title('Actions')
        self.ax_actions.set_xlabel('Episode')
        self.ax_actions.set_ylabel('Action Value')
        self.ax_actions.legend()
        self.ax_actions.grid(True)

        # 更新图表
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            logging.warning(f"Error updating training plots: {e}")

    def save_training_plots(self):
        """保存训练过程的图表"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存训练进度图
        self.fig.savefig(self.log_dir / f'training_progress_{timestamp}.png')

        # 保存最后一帧的传感器数据图
        self.fig_sensors.savefig(self.log_dir / f'last_sensor_data_{timestamp}.png')

        plt.close(self.fig)
        plt.close(self.fig_sensors)
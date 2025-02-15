import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm
import torch.nn.functional as F
from carla_env import CarlaEnv
from model import AutoDrivingNetwork, PPOMemory
from visualization import Visualizer


class PPOTrainer:
    """PPO算法训练器"""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 创建环境
        self.env = CarlaEnv(config)

        # 创建模型
        self.model = AutoDrivingNetwork(config)

        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.ppo_params['learning_rate']
        )

        # 创建经验回放缓冲区
        self.memory = PPOMemory(config)

        # 创建可视化工具
        self.visualizer = Visualizer(
            config.log.log_params['log_dir'],
            config
        )

        # 设置训练参数
        self.n_epochs = config.training.ppo_params['n_epochs']
        self.batch_size = config.training.ppo_params['batch_size']
        self.clip_range = config.training.ppo_params['clip_range']
        self.value_coef = config.training.ppo_params['value_coef']
        self.entropy_coef = config.training.ppo_params['entropy_coef']
        self.max_grad_norm = config.training.ppo_params['max_grad_norm']
        self.gamma = config.training.ppo_params['gamma']
        self.gae_lambda = config.training.ppo_params['gae_lambda']

        # 训练统计
        self.training_info = {
            'total_timesteps': 0,
            'episodes': 0,
            'start_time': time.time(),
            'best_eval_reward': float('-inf')
        }

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志记录"""
        log_dir = Path(self.config.log.log_params['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=getattr(logging, self.config.log.log_params['log_level']),
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def collect_rollouts(self, n_steps: int) -> Dict[str, torch.Tensor]:
        """收集训练数据"""
        obs = self.env.reset()[0]
        episode_reward = 0
        episode_length = 0
        rgb_images = []

        for step in range(n_steps):
            # 转换观察为张量
            obs_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
            }

            # 获取动作
            with torch.no_grad():
                action, action_info = self.model.get_action(obs_tensor)

            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action.cpu().numpy()[0])

            # 更新可视化
            if self.visualizer is not None:
                self.visualizer.visualize_sensor_data(obs['rgb'], obs['lidar'])

                if done or truncated:
                    self.visualizer.update_training_info(
                        self.training_info['episodes'],
                        episode_reward,
                        info.get('reward_info', {}),
                        action.cpu().numpy()[0]
                    )

            # 记录数据
            self.memory.add(
                obs_tensor,
                action,
                reward,
                action_info['value'],
                action_info['log_prob'],
                float(not (done or truncated))
            )

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            if done or truncated:
                self.training_info['episodes'] += 1
                self.logger.info(
                    f"Episode {self.training_info['episodes']}: "
                    f"reward={episode_reward:.2f}, length={episode_length}"
                )
                obs = self.env.reset()[0]
                episode_reward = 0
                episode_length = 0
                rgb_images = []

        return self.memory.get_batch()

    def train_epoch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练一个epoch"""
        # 计算优势估计和回报
        advantages = torch.zeros_like(batch['rewards'])
        returns = torch.zeros_like(batch['rewards'])

        last_gae = 0
        for t in reversed(range(len(batch['rewards']))):
            if t == len(batch['rewards']) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = batch['values'][t + 1]
                next_non_terminal = batch['dones'][t + 1]

            delta = (batch['rewards'][t] +
                     self.gamma * next_value * next_non_terminal -
                     batch['values'][t])

            last_gae = (delta +
                        self.gamma * self.gae_lambda *
                        next_non_terminal * last_gae)

            advantages[t] = last_gae
            returns[t] = advantages[t] + batch['values'][t]

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练多个epoch
        epoch_stats = []
        for _ in range(self.n_epochs):
            # 打乱数据
            indices = torch.randperm(len(returns))

            # 分批训练
            for start_idx in range(0, len(returns), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                # 评估动作
                new_log_prob, entropy, value = self.model.evaluate_actions(
                    {k: v[batch_indices] for k, v in batch['observations'].items()},
                    batch['actions'][batch_indices]
                )

                # 计算策略损失
                ratio = torch.exp(new_log_prob - batch['log_probs'][batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算值损失
                value_loss = F.mse_loss(value.squeeze(), returns[batch_indices])

                # 计算熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss)

                # 优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 记录统计信息
                epoch_stats.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'total_loss': loss.item(),
                    'approx_kl': (batch['log_probs'][batch_indices] - new_log_prob).mean().item()
                })

        # 计算平均统计信息
        mean_stats = {
            k: np.mean([s[k] for s in epoch_stats])
            for k in epoch_stats[0].keys()
        }

        return mean_stats

    def train(self):
        """训练模型"""
        total_timesteps = self.config.training.base_params['total_timesteps']
        n_steps = self.config.training.base_params['n_steps_per_episode']
        eval_interval = self.config.training.base_params['eval_interval']
        save_interval = self.config.training.base_params['save_interval']

        n_updates = total_timesteps // n_steps

        self.logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.logger.info(f"Number of updates: {n_updates}")

        for update in tqdm(range(1, n_updates + 1), desc="Training"):
            # 收集数据
            batch = self.collect_rollouts(n_steps)

            # 训练
            train_stats = self.train_epoch(batch)

            # 记录训练信息
            self.logger.info(
                f"Update {update}/{n_updates}: "
                f"policy_loss={train_stats['policy_loss']:.4f}, "
                f"value_loss={train_stats['value_loss']:.4f}, "
                f"entropy_loss={train_stats['entropy_loss']:.4f}, "
                f"approx_kl={train_stats['approx_kl']:.4f}"
            )

            # 定期评估
            if update % (eval_interval // n_steps) == 0:
                eval_reward = self.evaluate()
                self.visualizer.update_eval_info(
                    self.training_info['total_timesteps'],
                    eval_reward
                )

                # 保存最佳模型
                if eval_reward > self.training_info['best_eval_reward']:
                    self.training_info['best_eval_reward'] = eval_reward
                    self.save_checkpoint('best_model.pt')

            # 定期保存模型
            if update % (save_interval // n_steps) == 0:
                self.save_checkpoint(f"checkpoint_{update}.pt")

        self.logger.info("Training completed!")
        self.save_checkpoint("final_model.pt")
        self.visualizer.save_training_plots()

    def evaluate(self, n_episodes: Optional[int] = None) -> float:
        """评估模型"""
        if n_episodes is None:
            n_episodes = self.config.log.eval_params['n_eval_episodes']

        self.model.eval()
        rewards = []

        for episode in range(n_episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            rgb_images = []

            while True:
                obs_tensor = {
                    k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                    for k, v in obs.items()
                }

                with torch.no_grad():
                    action, _ = self.model.get_action(obs_tensor, deterministic=True)

                obs, reward, done, truncated, _ = self.env.step(action.cpu().numpy()[0])
                episode_reward += reward

                if self.config.log.eval_params['record_video']:
                    rgb_images.append(obs['rgb'])

                if done or truncated:
                    break

            rewards.append(episode_reward)

            # 保存评估视频
            if self.config.log.eval_params['record_video']:
                video_path = (Path(self.config.log.log_params['eval_video_dir']) /
                              f"eval_episode_{episode}.mp4")
                self.visualizer.record_video(rgb_images, str(video_path))

        self.model.train()
        return np.mean(rewards)

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_dir = Path(self.config.log.log_params['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_info': self.training_info,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_dir / filename)
        self.logger.info(f"Saved checkpoint to {checkpoint_dir / filename}")

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint_path = Path(self.config.log.log_params['checkpoint_dir']) / filename

        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint {checkpoint_path} does not exist!")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_info = checkpoint['training_info']

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")


def main():
    """主训练脚本"""
    # 加载配置
    from config import Config
    config = Config()

    # 创建训练器
    trainer = PPOTrainer(config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
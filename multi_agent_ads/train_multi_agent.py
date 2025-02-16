"""
File: train_multi_agent.py

Description:
  An example training script for multi-agent RL using:
    - carla_pettingzoo_env.py (PettingZoo ParallelEnv in CARLA)
    - multi_agent_rl_agent.py (Independent DDPG multi-agent implementation)

Steps:
  1) Initialize environment (CarlaPettingZooEnv).
  2) Initialize MultiAgentRLAgent (with or without pre-trained weights).
  3) For each episode:
       - reset env, get initial observations
       - roll out the episode until done (or truncated)
         * agent.select_actions(obs)
         * env.step(actions)
         * agent.store_transition(...)
         * agent.update()
       - logging & periodic saving

Usage:
  python train_multi_agent.py --episodes 100 --save_interval 10 ...
"""

import argparse
import numpy as np
import os
import time

# 1) 引入我们之前实现的 环境 与 智能体类
from carla_pettingzoo_env import CarlaPettingZooEnv
from multi_agent_rl_agent import MultiAgentRLAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=3, help="Number of vehicles/agents in CARLA.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps (per episode) in env.")
    parser.add_argument("--obs_dim", type=int, default=8, help="Dimension of each agent's observation.")
    parser.add_argument("--action_dim", type=int, default=3, help="Dimension of each agent's action.")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="Actor network learning rate.")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Critic network learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update factor for target networks.")
    parser.add_argument("--hidden_dims", type=str, default="128,128",
                        help="Comma-separated hidden layer sizes for actor/critic networks.")
    parser.add_argument("--buffer_capacity", type=int, default=100000,
                        help="Replay buffer capacity.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--updates_per_step", type=int, default=1,
                        help="Number of gradient update steps per environment step.")
    parser.add_argument("--exploration_noise", type=float, default=0.1,
                        help="Std. dev. of Gaussian exploration noise.")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save model every N episodes.")
    parser.add_argument("--model_dir", type=str, default="./saved_models",
                        help="Directory to save or load model checkpoints.")
    parser.add_argument("--render_mode", type=str, default=None,
                        help="Optional: 'human' to visualize. None for no rendering.")
    parser.add_argument("--resume_step", type=int, default=0,
                        help="If >0, try loading existing model with that step index.")
    return parser.parse_args()


def main():
    args = parse_args()
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    # 2) 初始化环境 (PettingZoo并行环境)
    print("[INFO] Initializing CARLA PettingZoo Environment...")
    env = CarlaPettingZooEnv(
        host="localhost",
        port=2000,
        agent_count=args.num_agents,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        max_steps=args.max_steps,
        noise_std=0.0,  # We already add noise in agent if needed
        render_mode=args.render_mode
    )

    # 3) 初始化多智能体RL智能体
    print("[INFO] Initializing MultiAgentRLAgent...")
    agent = MultiAgentRLAgent(
        num_agents=args.num_agents,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        hidden_dims=hidden_dims,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        exploration_noise=args.exploration_noise,
        model_dir=args.model_dir
    )

    # 如果指定了 resume_step，则尝试加载已有的模型
    if args.resume_step > 0:
        print(f"[INFO] Loading model from step {args.resume_step} ...")
        agent.load(step=args.resume_step)

    # 4) 训练循环
    total_episodes = args.episodes
    for ep in range(1, total_episodes + 1):
        start_time = time.time()

        # reset environment
        observations = env.reset(seed=None)  # returns dict: {agent_id: obs_np, ...}
        # ParallelEnv reset returns (obs_dict, info_dict), so if we follow PettingZoo strictly:
        # observations, infos = env.reset()
        # 但本示例中 CarlaPettingZooEnv.reset()返回 (obs, {})

        # 一些统计量
        episode_reward_sum = {agent_id: 0.0 for agent_id in env.agents}
        done_flags = {agent_id: False for agent_id in env.agents}
        truncated_flags = {agent_id: False for agent_id in env.agents}
        step_count = 0

        while True:
            step_count += 1
            # 选择动作 (dict形式)
            actions = agent.select_actions(observations)

            # 环境执行
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # 将当前transition存储到回放池
            # 需要把 obs, act, rew, next_obs, done 整理为多智能体数组
            agent.store_transition(
                obs_dict=observations,
                act_dict=actions,
                rew_dict=rewards,
                next_obs_dict=next_observations,
                done_dict=terminations
            )

            # 调用更新：可根据需求每步或每几步更新
            agent.update(updates_per_step=args.updates_per_step)

            # 累积奖励
            for agent_id in env.agents:
                episode_reward_sum[agent_id] += rewards[agent_id]

            # 更新 obs
            observations = next_observations

            # 更新 done/truncated标志
            for agent_id in env.agents:
                done_flags[agent_id] = terminations[agent_id]
                truncated_flags[agent_id] = truncations[agent_id]

            # 判断是否全部结束
            # PettingZoo并行模式下，只要有一个agent done并不一定要结束，也可自定义
            # 这里示例：只要全部智能体都 done 或 truncated，我们就结束回合
            all_done = all(done_flags[a] or truncated_flags[a] for a in env.agents)
            if all_done:
                break

        # 回合结束，统计结果
        ep_duration = time.time() - start_time
        mean_ep_reward = np.mean(list(episode_reward_sum.values()))
        print(f"[TRAIN] Episode {ep}/{total_episodes} finished: Steps={step_count}, "
              f"MeanReward={mean_ep_reward:.3f}, Duration={ep_duration:.2f}s")

        # 定期保存
        if ep % args.save_interval == 0:
            agent.save(step=ep)

    # 训练结束，保存最终模型
    agent.save(step=total_episodes)
    env.close()
    print("[INFO] Training complete. Environment closed.")


if __name__ == "__main__":
    main()

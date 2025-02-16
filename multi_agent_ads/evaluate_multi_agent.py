"""
File: evaluate_multi_agent.py

Description:
  A script to evaluate a trained multi-agent RL model in the CarlaPettingZooEnv.
  It:
    1) Loads the saved model checkpoints from a specified step (or final step).
    2) Runs a specified number of evaluation episodes (no training).
    3) Logs or prints average rewards, collision rates, or other metrics.
    4) (Optional) Can render or record videos for qualitative analysis.

Usage Example:
  python evaluate_multi_agent.py --episodes 10 --load_step 100 --render_mode human
"""

import argparse
import numpy as np
import time
import os

# 1) Import the environment and the multi-agent RL agent class
from carla_pettingzoo_env import CarlaPettingZooEnv
from multi_agent_rl_agent import MultiAgentRLAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents in CARLA.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps (per episode).")
    parser.add_argument("--obs_dim", type=int, default=8, help="Dimension of each agent's observation.")
    parser.add_argument("--action_dim", type=int, default=3, help="Dimension of each agent's action.")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="Actor network learning rate.")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Critic network learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update factor for target nets.")
    parser.add_argument("--hidden_dims", type=str, default="128,128",
                        help="Comma-separated hidden layer sizes for networks.")
    parser.add_argument("--model_dir", type=str, default="./saved_models",
                        help="Where the trained model is stored.")
    parser.add_argument("--load_step", type=int, default=0,
                        help="Which step (checkpoint) to load. If 0, tries final or defaults.")
    parser.add_argument("--render_mode", type=str, default=None,
                        help="Set to 'human' to visualize the simulation.")
    parser.add_argument("--episodes_delay", type=float, default=0.0,
                        help="Optional delay (seconds) between episodes for clarity or logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    # 2) Initialize environment in evaluation mode (render_mode if you want to see the simulation)
    print("[INFO] Initializing CARLA environment for evaluation...")
    env = CarlaPettingZooEnv(
        host="localhost",
        port=2000,
        num_agents=args.num_agents,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        max_steps=args.max_steps,
        noise_std=0.0,           # Typically we don't add noise in evaluation
        render_mode=args.render_mode
    )

    # 3) Initialize the multi-agent RL agent structure
    #    Even though we're only evaluating, we need the same architecture.
    print("[INFO] Initializing MultiAgentRLAgent for evaluation...")
    agent = MultiAgentRLAgent(
        num_agents=args.num_agents,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        hidden_dims=hidden_dims,
        buffer_capacity=1,   # buffer not really needed in eval, but must not be zero
        batch_size=1,
        exploration_noise=0.0,  # no exploration noise in evaluation
        model_dir=args.model_dir
    )

    # 4) Load the trained model (if available)
    if args.load_step > 0:
        print(f"[INFO] Loading from step {args.load_step}")
        agent.load(step=args.load_step)
    else:
        # If load_step=0, we can attempt to load from the final or fallback
        # Or skip if not found
        checkpoint_candidates = sorted([
            f for f in os.listdir(args.model_dir)
            if f.startswith("agent_0_step_") and f.endswith(".pth")
        ])
        if len(checkpoint_candidates) == 0:
            print("[WARNING] No model found in model_dir; using untrained policy.")
        else:
            # e.g. pick the last one
            last_ckpt = checkpoint_candidates[-1]
            step_str = last_ckpt[len("agent_0_step_") : -len(".pth")]  # parse step number
            try:
                last_step = int(step_str)
                print(f"[INFO] No --load_step provided. Auto-loading last checkpoint step={last_step}.")
                agent.load(step=last_step)
            except:
                print(f"[WARNING] Could not parse step from {last_ckpt}. Using untrained policy.")

    # 5) Evaluation loop
    total_episodes = args.episodes
    all_episode_rewards = []
    all_collisions_count = []

    for ep in range(1, total_episodes + 1):
        start_time = time.time()
        obs = env.reset()
        # If PettingZoo returns (obs, info), do: obs, info = env.reset()

        episode_reward_sum = {agent_id: 0.0 for agent_id in env.agents}
        done_flags = {agent_id: False for agent_id in env.agents}
        truncated_flags = {agent_id: False for agent_id in env.agents}

        step_count = 0
        collisions_count = 0

        while True:
            step_count += 1
            # agent selects action (no exploration noise in eval)
            actions = agent.select_actions(obs)  # returns dict {agent_0: action, ...}
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # accumulate rewards
            for agent_id in env.agents:
                episode_reward_sum[agent_id] += rewards[agent_id]
                # check collisions in info or environment
                # e.g. if "collision" in infos[agent_id], or handle PettingZoo's own collision logic
                if "collision" in infos[agent_id]:
                    if infos[agent_id]["collision"] is True:
                        collisions_count += 1

            obs = next_obs

            # check done
            for agent_id in env.agents:
                done_flags[agent_id] = terminations[agent_id]
                truncated_flags[agent_id] = truncations[agent_id]

            all_done = all(done_flags[a] or truncated_flags[a] for a in env.agents)
            if all_done:
                break

        ep_duration = time.time() - start_time
        ep_rewards = list(episode_reward_sum.values())  # each agent's total reward
        mean_ep_reward = np.mean(ep_rewards)

        all_episode_rewards.append(mean_ep_reward)
        all_collisions_count.append(collisions_count)

        print(f"[EVAL] Episode {ep}/{total_episodes} "
              f"Steps={step_count}, Duration={ep_duration:.2f}s, "
              f"MeanReward={mean_ep_reward:.3f}, Collisions={collisions_count}")

        # optional small delay between episodes
        if args.episodes_delay > 0:
            time.sleep(args.episodes_delay)

    # 6) Final statistics
    mean_reward_all_episodes = np.mean(all_episode_rewards)
    mean_collisions = np.mean(all_collisions_count)
    print("===============================================")
    print(f"[RESULT] Evaluation Done. Episodes: {total_episodes}")
    print(f" - Average Episode Reward: {mean_reward_all_episodes:.3f}")
    print(f" - Average Collisions per Episode: {mean_collisions:.2f}")
    print("===============================================")

    # close environment
    env.close()
    print("[INFO] Environment closed. Evaluation finished.")


if __name__ == "__main__":
    main()

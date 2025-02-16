"""
File: multi_agent_rl_agent.py

Description:
  A more complete Multi-Agent RL Agent implementation using PyTorch,
  based on the Independent DDPG approach for continuous control.
  - Each agent has its own Actor-Critic network, target networks, and optimizers.
  - A global ReplayBuffer is shared among all agents to store transitions.
  - Off-policy training with soft target updates.
  - select_actions(): run forward pass in each agent's actor to get continuous actions.
  - store_transition(): store multi-agent transitions into replay buffer.
  - update(): sample from replay buffer, do one or multiple gradient steps for each agent.
  - save() / load(): save or load all agents' model weights.

Usage Example (in your training script):
  from agents.multi_agent_rl_agent import MultiAgentRLAgent

  # Example instantiation
  agent = MultiAgentRLAgent(
      num_agents=3,
      obs_dim=8,
      action_dim=3,
      actor_lr=1e-3,
      critic_lr=1e-3,
      gamma=0.99,
      tau=0.01,
      buffer_capacity=100000,
      batch_size=128
  )

  # In training loop:
  observations = env.reset()
  while not done:
      actions = agent.select_actions(observations)
      next_observations, rewards, dones, truncations, info = env.step(actions)
      agent.store_transition(observations, actions, rewards, next_observations, dones)
      agent.update()  # gradient step (can be called less/more frequently)
      observations = next_observations

  agent.save(step=global_training_step)

Note:
  - For real multi-agent algorithms like MADDPG, you'd design the critics to observe global states
    and all agents' actions, but here we show a simpler "Independent DDPG" version for demonstration.
  - Adjust hyperparameters, model architecture, etc. for your research demands.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===================================================================
# 1) Utility: Replay Buffer
# ===================================================================

class ReplayBuffer:
    """
    A simple global replay buffer to store multi-agent transitions:
      - obs, actions, rewards, next_obs, done
    For each transition, we store info for all agents at once.

    Storage shape (when num_agents > 1):
      obs[i] shape = (num_agents, obs_dim)
      act[i] shape = (num_agents, action_dim)
      rew[i] shape = (num_agents,)
      next_obs[i] shape = (num_agents, obs_dim)
      done[i] shape = (num_agents,)

    We sample batches in a vectorized form and then feed them to each agent's update.
    """

    def __init__(self, obs_dim, action_dim, num_agents, capacity=100000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        # Create buffers
        # shape: (capacity, num_agents, obs_dim)
        self.obs_buf = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, num_agents), dtype=np.float32)
        self.done_buf = np.zeros((capacity, num_agents), dtype=np.float32)

    def store_transition(self, obs, act, rew, next_obs, done):
        """
        obs: dict {agent_0: obs_0, ...} or a multi-agent array
        act: dict or multi-agent array
        rew: dict or multi-agent array
        next_obs: dict or multi-agent array
        done: dict or multi-agent array
        We assume the caller has already formed them in shape [num_agents, ...].
        """

        idx = self.ptr
        self.obs_buf[idx] = obs  # shape (num_agents, obs_dim)
        self.act_buf[idx] = act  # shape (num_agents, action_dim)
        self.rew_buf[idx] = rew  # shape (num_agents,)
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = done

        # move pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        """
        Return a batch of transitions randomly sampled from replay buffer.
        Each item is shape: (batch_size, num_agents, ...)
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs]
        )
        return batch

    def __len__(self):
        return self.size


# ===================================================================
# 2) Utility: Neural Network Models
# ===================================================================

def mlp_block(input_dim, hidden_dims=(128,128), output_dim=1, activation=nn.ReLU, final_activation=None):
    """
    A helper function to build an MLP.
    """
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(activation())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Actor network for DDPG, outputs continuous actions in [-1,1].
    We'll use a tanh on the final layer to clamp the outputs.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=(128,128)):
        super().__init__()
        self.net = mlp_block(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            final_activation=nn.Tanh  # clamp outputs to [-1,1]
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """
    Critic network for DDPG/Q-learning:
    Input: concat of obs and action
    Output: Q-value (scalar)
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=(128,128)):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.net = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            final_activation=None
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# ===================================================================
# 3) Class: IndependentDDPGAgent
# ===================================================================

class IndependentDDPGAgent:
    """
    A single-agent DDPG component used within a multi-agent system.
    It has:
      - Actor, Critic
      - Target Actor, Target Critic
      - Optimizers for Actor, Critic
    """

    def __init__(self, obs_dim, action_dim, actor_lr, critic_lr, gamma, tau, hidden_dims=(128,128)):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor & Critic
        self.actor = Actor(obs_dim, action_dim, hidden_dims)
        self.critic = Critic(obs_dim, action_dim, hidden_dims)

        # Target networks
        self.actor_target = Actor(obs_dim, action_dim, hidden_dims)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dims)

        # Copy initial weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, obs_np, noise_std=0.0):
        """
        obs_np: shape (obs_dim,) in numpy
        Return continuous action in [-1,1], shape (action_dim,)
        If noise_std > 0, add Gaussian noise for exploration.
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            act = self.actor(obs)
        act = act.squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            act += np.random.randn(self.action_dim) * noise_std
        return np.clip(act, -1.0, 1.0)

    def update(self, obs, act, rew, next_obs, done,):
        """
        Perform one gradient step of DDPG for this agent.
        obs, act, rew, next_obs, done shape: (batch_size, obs_dim or action_dim)
        """

        # to tensor
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # ------------------ update critic ------------------ #
        with torch.no_grad():
            # target actions
            next_act_t = self.actor_target(next_obs_t)
            # target Q
            target_q = self.critic_target(next_obs_t, next_act_t)
            target_q = rew_t + self.gamma * (1 - done_t) * target_q

        current_q = self.critic(obs_t, act_t)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------ update actor ------------------ #
        # policy gradient: maximize Q(obs, actor(obs))
        # or equivalently minimize negative Q
        predicted_act = self.actor(obs_t)
        actor_loss = -self.critic(obs_t, predicted_act).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------ soft update target networks ------------------ #
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, net, net_target):
        for param, param_targ in zip(net.parameters(), net_target.parameters()):
            param_targ.data.copy_(self.tau * param.data + (1 - self.tau) * param_targ.data)

    def save(self, path):
        """
        Save this agent's parameters.
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        """
        Load this agent's parameters.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ===================================================================
# 4) Main: MultiAgentRLAgent (manages multiple IndependentDDPGAgent)
# ===================================================================

class MultiAgentRLAgent:
    """
    A multi-agent RL wrapper that manages multiple IndependentDDPGAgent instances,
    plus a global replay buffer.
    - select_actions() to get all agents' actions
    - store_transition() to push multi-agent data into buffer
    - update() to do the gradient steps for each agent
    - save() / load() to handle all agents
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
        hidden_dims=(128, 128),
        buffer_capacity=100000,
        batch_size=128,
        exploration_noise=0.1,
        model_dir="./saved_models"
    ):
        """
        :param num_agents: Number of agents
        :param obs_dim: dimension of each agent's observation
        :param action_dim: dimension of each agent's continuous action
        :param actor_lr: Actor learning rate
        :param critic_lr: Critic learning rate
        :param gamma: discount factor
        :param tau: soft update factor
        :param hidden_dims: hidden layer sizes for Actor/Critic networks
        :param buffer_capacity: max size of replay buffer
        :param batch_size: number of samples per training batch
        :param exploration_noise: std for exploration noise
        :param model_dir: where to save / load the model
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.model_dir = model_dir

        # Create individual DDPG agents
        self.agents = [
            IndependentDDPGAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                hidden_dims=hidden_dims
            ) for _ in range(num_agents)
        ]

        # Global replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            capacity=buffer_capacity
        )

        # Create model directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

    def select_actions(self, observations_dict):
        """
        observations_dict: { "agent_0": obs0, "agent_1": obs1, ... }
        or a similar structure. Must transform them to shape [num_agents, obs_dim].
        Return a dict of actions: { "agent_0": act0, ... } in [-1,1].
        """
        # If observations_dict is dict, we can read each agent's obs
        actions_dict = {}
        for i, (agent_id, obs) in enumerate(observations_dict.items()):
            # obs shape = (obs_dim,)
            action = self.agents[i].select_action(obs, noise_std=self.exploration_noise)
            actions_dict[agent_id] = action
        return actions_dict

    def store_transition(self, obs_dict, act_dict, rew_dict, next_obs_dict, done_dict):
        """
        Transform the input from dict form into arrays and store in replay buffer.
        obs_dict:  { "agent_0": obs0, ... } shape = (obs_dim,)
        act_dict:  { "agent_0": act0, ... } shape = (action_dim,)
        rew_dict:  { "agent_0": rew0, ... } float
        next_obs_dict: ...
        done_dict:  { "agent_0": bool, ... }

        We'll store them in the shape [num_agents, ...].
        """
        obs_arr = []
        act_arr = []
        rew_arr = []
        next_obs_arr = []
        done_arr = []

        # Sort by agent index to ensure consistent ordering
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            obs_arr.append(obs_dict[agent_id])
            act_arr.append(act_dict[agent_id])
            rew_arr.append(rew_dict[agent_id])
            next_obs_arr.append(next_obs_dict[agent_id])
            done_arr.append(float(done_dict[agent_id]))

        obs_arr = np.array(obs_arr, dtype=np.float32)       # shape (num_agents, obs_dim)
        act_arr = np.array(act_arr, dtype=np.float32)       # shape (num_agents, action_dim)
        rew_arr = np.array(rew_arr, dtype=np.float32)       # shape (num_agents,)
        next_obs_arr = np.array(next_obs_arr, dtype=np.float32)
        done_arr = np.array(done_arr, dtype=np.float32)     # shape (num_agents,)

        # store
        self.replay_buffer.store_transition(obs_arr, act_arr, rew_arr, next_obs_arr, done_arr)

    def update(self, updates_per_step=1):
        """
        Sample a batch from the replay buffer and update each agent's network.
        :param updates_per_step: how many gradient steps to run per call
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data

        for _ in range(updates_per_step):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            # batch["obs"] shape = (batch_size, num_agents, obs_dim)
            # We'll train each agent independently using its portion of the data

            obs_all = batch["obs"]        # shape (B, N, obs_dim)
            actions_all = batch["act"]    # shape (B, N, action_dim)
            rewards_all = batch["rew"]    # shape (B, N)
            next_obs_all = batch["next_obs"]  # shape (B, N, obs_dim)
            done_all = batch["done"]      # shape (B, N)

            # Each agent trains with (obs[i], act[i], rew[i], next_obs[i], done[i])
            # This is "independent" version. (MADDPG would combine states/actions across agents in the critic.)
            for i in range(self.num_agents):
                obs_i = obs_all[:, i, :]       # shape (B, obs_dim)
                act_i = actions_all[:, i, :]   # shape (B, action_dim)
                rew_i = rewards_all[:, i]      # shape (B,)
                next_obs_i = next_obs_all[:, i, :]
                done_i = done_all[:, i]

                actor_loss, critic_loss = self.agents[i].update(obs_i, act_i, rew_i, next_obs_i, done_i)

    def save(self, step=0):
        """
        Save all agents' parameters.
        """
        for i in range(self.num_agents):
            save_path = os.path.join(self.model_dir, f"agent_{i}_step_{step}.pth")
            self.agents[i].save(save_path)
        print(f"[INFO] Saved model parameters at step {step} to {self.model_dir}")

    def load(self, step=0):
        """
        Load all agents' parameters from files.
        """
        for i in range(self.num_agents):
            load_path = os.path.join(self.model_dir, f"agent_{i}_step_{step}.pth")
            if os.path.exists(load_path):
                self.agents[i].load(load_path)
            else:
                print(f"[WARNING] No checkpoint found at {load_path}")

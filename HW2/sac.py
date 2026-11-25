"""
CS 557 RLI Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
import random
from collections import namedtuple, deque
from itertools import count
from logger import Logger
import utils


# Replay Buffer for off-policy learning (pre-allocated NumPy ring buffer for O(batch) sampling)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def _allocate(self, state_dim, action_dim):
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        s_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        a_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        ns_arr = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if self.states is None:
            self._allocate(s_arr.shape[0], a_arr.shape[0])

        self.states[self.ptr] = s_arr
        self.actions[self.ptr] = a_arr
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = ns_arr
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size



# Actor Network (Policy)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mean, std
    
    def deterministic_action(self, state):
        """
        Action used for evaluation.

        SAC squashes its actions through a tanh, so the deterministic action is
        the squashed mean rather than the raw mean. Using the raw mean here would
        evaluate the policy with a different action mapping than it trains with.
        """
        mean, _ = self.forward(state)
        return torch.tanh(mean)

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        # https://spinningup.openai.com/en/latest/algorithms/sac.html

        # HINT: rsample is a method that is used to sample from a distribution
        # in a way that allows for gradients to be backpropagated through the sampling process.
        # The reason that we use rsample instead of sample is that
        # sample does not allow for gradients to be backpropagated
        # but we need to be able to backpropagate through the action sampling,
        # in order to update the policy network in SAC.
        # HINT2: After sampling, apply the tanh function to the sampled action
        # to ensure the action is in the range [-1, 1].
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log_prob
        log_prob = normal.log_prob(x_t)
        
        # Apply correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)



# Critic Network (Q-Value)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.linear3(x))
        q2 = F.relu(self.linear4(q2))
        q2 = self.q2(q2)
        
        return q1, q2




class SACAgent:
    '''Soft Actor-Critic Agent for continuous action spaces'''

    def __init__(self, env_name, lr=3e-4, max_steps=1_000_000, batch_size=256, gamma=0.99,
                 tau=0.005, alpha=0.2, buffer_size=1000000, updates_per_step=1, hidden_size=256,
                 save_interval=30_000, start_steps=10000, ou_warmup=False, seed=None, eval_episodes=10):
        '''Initialize SAC agent'''

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.seed = seed
        self.eval_episodes = eval_episodes

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env.action_space.seed(seed)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action space: {self.env.action_space.low} to {self.env.action_space.high}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyper parameters
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps  # Exploration steps before using policy
        self.ou_warmup = ou_warmup
        self.save_interval = save_interval
        self.hidden_size = hidden_size

        self.action_low_np = self.env.action_space.low.astype(np.float32)
        self.action_high_np = self.env.action_space.high.astype(np.float32)

        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim=hidden_size).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim=hidden_size).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim=hidden_size).to(self.device)

        # Initialize target network with same weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        init_alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.log_alpha = torch.log(init_alpha.reshape(1)).detach().requires_grad_(True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.policy.parameters()) + sum(p.numel() for p in self.critic.parameters())
        self.variant_tag = "sac"
        hparams = {'lr': lr, 'gamma': gamma, 'tau': tau, 'bs': batch_size, 'buf': buffer_size}
        if ou_warmup:
            hparams['ou'] = 1
        self.logger = Logger(self.env_tag, self.env_name, self.variant_tag, self.num_params, hparams=hparams)

        # Step counter
        self.total_steps = 0
        self._last_q_mean = 0.0
        self._last_entropy = 0.0

    def _scale_action(self, action_norm):
        return self.action_low_np + 0.5 * (action_norm + 1.0) * (self.action_high_np - self.action_low_np)

    def _unscale_action(self, action_env):
        return 2.0 * (action_env - self.action_low_np) / (self.action_high_np - self.action_low_np + 1e-8) - 1.0

    def select_action(self, state):
        '''Select action in [-1, 1] according to current policy'''
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, _, _ = self.policy.sample(state_t)
            return action.cpu().numpy()[0]

    def update_parameters(self):
        '''Update the networks using SAC update rules'''

        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        state_batch = torch.from_numpy(state_batch).to(self.device)
        action_batch = torch.from_numpy(action_batch).to(self.device)
        reward_batch = torch.from_numpy(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.from_numpy(next_state_batch).to(self.device)
        done_batch = torch.from_numpy(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1.0 - done_batch) * self.gamma * target_q

        # Critic update
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy update (detach alpha so policy_loss does not backprop into log_alpha)
        new_actions, log_prob, _ = self.policy.sample(state_batch)
        q1, q2 = self.critic(state_batch, new_actions)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha.detach() * log_prob - q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature alpha update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        self._last_q_mean = float(q.mean().item())
        self._last_entropy = float(-log_prob.mean().item())
        return critic_loss.item(), policy_loss.item(), alpha_loss.item()

    def _save_checkpoint(self):
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/{self.logger.run_name}_{self.env_name}_steps_{self.total_steps}.pth'
        torch.save(self.policy.state_dict(), ckpt_path)
        return ckpt_path

    def train_sac(self):
        """
        Train the agent using Soft Actor-Critic algorithm with logging
        """

        i_episode = 0
        last_save = -self.save_interval
        action_dim = self.env.action_space.shape[0]

        while self.total_steps < self.max_steps:
            reset_seed = self.seed if (self.seed is not None and i_episode == 0) else None
            state, _ = self.env.reset(seed=reset_seed)
            episode_reward = 0.0
            episode_steps = 0
            done = False
            ou_state = np.zeros(action_dim, dtype=np.float32)

            critic_loss = 0.0
            policy_loss = 0.0
            alpha_loss = 0.0

            while not done:
                if self.total_steps < self.start_steps:
                    if self.ou_warmup:
                        ou_state = 0.95 * ou_state + 0.3 * np.random.randn(action_dim).astype(np.float32)
                        action_norm = np.clip(ou_state, -1.0, 1.0)
                    else:
                        action_env = self.env.action_space.sample()
                        action_norm = self._unscale_action(action_env)
                else:
                    action_norm = self.select_action(state)

                action_env = np.clip(self._scale_action(action_norm), self.action_low_np, self.action_high_np)

                next_state, reward, terminated, truncated, _ = self.env.step(action_env)
                done = terminated or truncated
                episode_steps += 1
                self.total_steps += 1
                episode_reward += reward

                # Store terminated (NOT truncated) so time-limit cutoffs bootstrap correctly
                self.memory.push(state, action_norm, reward, next_state, float(terminated))
                state = next_state

                if self.total_steps >= self.start_steps and len(self.memory) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        critic_loss, policy_loss, alpha_loss = self.update_parameters()

            # Log episode statistics against environment steps
            self.logger.add_value('train/episode_reward', episode_reward, self.total_steps)
            self.logger.add_value('train/episode_length', episode_steps, self.total_steps)
            self.logger.add_value('train/critic_loss', critic_loss, self.total_steps)
            self.logger.add_value('train/policy_loss', policy_loss, self.total_steps)
            self.logger.add_value('train/alpha_loss', alpha_loss, self.total_steps)
            self.logger.add_value('train/alpha', self.alpha.item(), self.total_steps)
            self.logger.add_value('train/q_mean', self._last_q_mean, self.total_steps)
            self.logger.add_value('train/policy_entropy', self._last_entropy, self.total_steps)
            self.logger.add_value('train/episode', i_episode, self.total_steps)

            if self.total_steps - last_save >= self.save_interval:
                last_save = self.total_steps
                self._save_checkpoint()

                avg_eval_reward, frames = utils.evaluate_policy(
                    self.policy, self.env.observation_space, self.env_name,
                    episodes=self.eval_episodes, return_frames=True
                )
                self.logger.add_value('train/eval_mean_reward', avg_eval_reward, self.total_steps)
                self.logger.add_frames('train/eval_video', frames, self.total_steps)

            print(f"Episode {i_episode} - Steps: {self.total_steps}/{self.max_steps} - Total Reward: {episode_reward:.2f} - Critic Loss: {critic_loss:.4f} - Policy Loss: {policy_loss:.4f}")
            i_episode += 1

        # Save final checkpoint and run final deterministic evaluation
        self._save_checkpoint()
        avg_eval_reward = utils.evaluate_policy(
            self.policy, self.env.observation_space, self.env_name,
            episodes=self.eval_episodes, return_frames=False
        )
        self.logger.add_value('train/eval_mean_reward', avg_eval_reward, self.total_steps)

        print('Training complete')
        self.logger.close()
        self.env.close()





"""
CS 557 RLI Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import gymnasium as gym
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from logger import Logger

# Range that the policy's log standard deviation is kept in
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0



class MLPNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_init=-0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) + log_std_init)

    def forward(self, state):
        network_features = self.net(state)
        mean = self.mean(network_features)
        std = torch.exp(self.log_std)
        return mean, std

    def deterministic_action(self, state):
        """Action used for evaluation: the mean of the action distribution"""
        mean, _ = self.forward(state)
        return mean

    def clamp_log_std(self):
        """
        Constrain std for stable actions.

        This is done after the optimizer step rather than with a clamp() in
        forward(), because clamp() has zero gradient outside of its bounds and
        would freeze log_std at a bound permanently once it drifted past it.
        """

        with torch.no_grad():
            self.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

    

class REINFORCEAgent:
    '''REINFORCE Agent on continuous action spaces'''

    def __init__(self, env_name, lr=1e-3, max_steps=1_000_000, gamma=0.99,
                 log_std_init=-0.5, save_interval=30_000, seed=None, eval_episodes=10):
        '''Initialize REINFORCE agent'''

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

        # hyper parameters
        self.lr = lr
        self.max_steps = max_steps
        self.gamma = gamma
        self.log_std_init = log_std_init
        self.save_interval = save_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize policy network
        self.policy = MLPNetwork(state_dim, action_dim, log_std_init=log_std_init).to(self.device)
        # initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Action bounds used to scale the sampled actions
        self.action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)

        # Total environment steps taken, used as the x axis for logging so that
        # curves are comparable across agents that update at different rates
        self.total_steps = 0

        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.policy.parameters())
        self.variant_tag = "reinforce"
        self.logger = Logger(self.env_tag, self.env_name, self.variant_tag, self.num_params,
                             hparams={'lr': lr, 'gamma': gamma, 'lsi': log_std_init})



    def select_action(self, state):
        '''Select action according to current policy'''

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        # Forward pass through the policy network to obtain Gaussian parameters
        mean, std = self.policy(state_tensor)

        # Sample raw action from Gaussian policy and compute joint log-probability
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)

        # Scale action to the environment's action space and clip to valid bounds
        scaled_action = self.action_low + (0.5 * (raw_action + 1.0) * (self.action_high - self.action_low))
        action = np.clip(scaled_action.detach().cpu().numpy(), self.env.action_space.low, self.env.action_space.high)

        return action, log_prob
        



    def calculate_discounted_returns(self, rewards):
        '''Calculate discounted Monte Carlo returns G_t backwards through the episode'''

        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        return returns



    def train_reinforce(self):
        """
        Train the agent using REINFORCE algorithm with logging
        """

        # Training loop, budgeted in environment steps rather than episodes so that
        # runs of different agents cover the same amount of experience
        i_episode = 0
        last_save = -self.save_interval
        os.makedirs('checkpoints', exist_ok=True)

        while self.total_steps < self.max_steps:
            reset_seed = self.seed + i_episode if (self.seed is not None and i_episode == 0) else None
            state, _ = self.env.reset(seed=reset_seed)
            log_probs, rewards = [], []
            total_reward = 0.0
            episode_steps = 0

            # Episode loop
            done = False
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                episode_steps += 1
                state = next_state
                self.total_steps += 1

            # Calculate returns and update policy
            discounted_returns = self.calculate_discounted_returns(rewards)
            returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)

            # std() is undefined for a single step, which would make the returns nan
            if len(discounted_returns) > 1:
                returns_normalized = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            else:
                returns_normalized = returns_tensor

            # REINFORCE policy loss: negative sum of log probabilities weighted by normalized returns
            policy_loss = -(torch.stack(log_probs) * returns_normalized).sum()

            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=float('inf')).item()
            self.optimizer.step()
            self.policy.clamp_log_std()

            # Log episode statistics against environment steps
            self.logger.add_value('train/episode_reward', total_reward, self.total_steps)
            self.logger.add_value('train/policy_loss', policy_loss.item(), self.total_steps)
            self.logger.add_value('train/episode_length', episode_steps, self.total_steps)
            self.logger.add_value('train/policy_std', torch.exp(self.policy.log_std).mean().item(), self.total_steps)
            self.logger.add_value('train/grad_norm', grad_norm, self.total_steps)
            self.logger.add_value('train/episode', i_episode, self.total_steps)

            # Save model and evaluate at intervals
            if self.total_steps - last_save >= self.save_interval:
                last_save = self.total_steps
                ckpt_path = f'checkpoints/{self.logger.run_name}_{self.env_name}_steps_{self.total_steps}.pth'
                torch.save(self.policy.state_dict(), ckpt_path)

                avg_eval_reward, frames = utils.evaluate_policy(
                    self.policy, self.env.observation_space, self.env_name,
                    episodes=self.eval_episodes, return_frames=True
                )
                self.logger.add_value('train/eval_mean_reward', avg_eval_reward, self.total_steps)
                self.logger.add_frames('train/eval_video', frames, self.total_steps)

            print(f"Episode {i_episode} - Steps: {self.total_steps}/{self.max_steps} - Total Reward: {total_reward:.2f} - Policy Loss: {policy_loss.item():.4f}")
            i_episode += 1

        # Save final checkpoint and run final deterministic evaluation
        final_ckpt_path = f'checkpoints/{self.logger.run_name}_{self.env_name}_steps_{self.total_steps}.pth'
        torch.save(self.policy.state_dict(), final_ckpt_path)
        avg_eval_reward = utils.evaluate_policy(
            self.policy, self.env.observation_space, self.env_name,
            episodes=self.eval_episodes, return_frames=False
        )
        self.logger.add_value('train/eval_mean_reward', avg_eval_reward, self.total_steps)

        print('Training complete')
        self.logger.close()
        self.env.close()

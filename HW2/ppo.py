"""
CS 557 RLI Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
import random
from logger import Logger
import utils

# Range that the policy's log standard deviation is kept in
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningObsNorm(nn.Module):
    """Running mean/variance normalizer stored as buffers for checkpoint & eval parity."""
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32))

    @torch.no_grad()
    def update(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.mean.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def forward(self, obs):
        return torch.clamp((obs - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


class RunningRewardNorm:
    """Running return-std reward scaling for critic stability; raw env rewards remain unchanged."""
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.ret = 0.0
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def normalize(self, reward, done):
        self.ret = self.ret * self.gamma + float(reward)
        delta = self.ret - self.mean
        tot_count = self.count + 1.0
        self.mean = self.mean + delta / tot_count
        m2 = self.var * self.count + (delta ** 2) * (self.count / tot_count)
        self.var = m2 / tot_count
        self.count = tot_count
        if done:
            self.ret = 0.0
        return float(np.clip(reward / np.sqrt(self.var + 1e-8), -10.0, 10.0))


# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_init=-0.5, norm_obs=False, use_sde=False, use_tanh=False):
        super().__init__()
        self.norm_obs = norm_obs
        self.use_sde = use_sde
        self.obs_norm = RunningObsNorm(state_dim) if norm_obs else None

        act_fn = nn.Tanh if (use_sde or use_tanh) else nn.ReLU
        self.network = nn.Sequential(
            _layer_init(nn.Linear(state_dim, 256)),
            act_fn(),
            _layer_init(nn.Linear(256, 256)),
            act_fn(),
        )

        # Policy head (actor)
        self.mean = _layer_init(nn.Linear(256, action_dim), std=0.01)
        if use_sde:
            self.log_std = nn.Parameter(torch.full((256, action_dim), log_std_init))
            self.register_buffer("exploration_mat", torch.zeros(256, action_dim))
            self.reset_noise()
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim) + log_std_init)

    def _preprocess(self, state):
        return self.obs_norm(state) if self.obs_norm is not None else state

    def update_obs_norm(self, obs):
        if self.obs_norm is not None:
            self.obs_norm.update(obs)

    def reset_noise(self):
        if self.use_sde:
            with torch.no_grad():
                std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
                self.exploration_mat.copy_(torch.randn_like(std) * std)

    def forward_features(self, state):
        return self.network(self._preprocess(state))

    def forward(self, state):
        features = self.forward_features(state)
        mean = self.mean(features)
        if self.use_sde:
            std_mat = torch.exp(self.log_std)
            var = features.pow(2) @ std_mat.pow(2) + 1e-6
            std = torch.sqrt(var)
        else:
            std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        features = self.forward_features(state)
        mean = self.mean(features)
        if self.use_sde:
            std_mat = torch.exp(self.log_std)
            var = features.pow(2) @ std_mat.pow(2) + 1e-6
            std = torch.sqrt(var)
            raw_action = mean + features @ self.exploration_mat
        else:
            std = torch.exp(self.log_std)
            raw_action = torch.distributions.Normal(mean, std).sample()
        return raw_action, mean, std

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
        The entropy bonus pushes log_std up every update, which is why we do it here.
        """
        with torch.no_grad():
            self.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)


# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, norm_obs=False, use_tanh=False):
        super().__init__()
        self.norm_obs = norm_obs
        self.obs_norm = RunningObsNorm(state_dim) if norm_obs else None
        act_fn = nn.Tanh if use_tanh else nn.ReLU
        self.network = nn.Sequential(
            _layer_init(nn.Linear(state_dim, 256)),
            act_fn(),
            _layer_init(nn.Linear(256, 256)),
            act_fn(),
            _layer_init(nn.Linear(256, 1), std=1.0),
        )

    def update_obs_norm(self, obs):
        if self.obs_norm is not None:
            self.obs_norm.update(obs)

    def forward(self, state):
        x = self.obs_norm(state) if self.obs_norm is not None else state
        return self.network(x)




class PPOAgent:
    '''PPO Agent for continuous action spaces'''

    def __init__(self, env_name, lr=3e-4, max_steps=1_000_000,
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                 value_coef=0.5, entropy_coef=0.01,
                 update_epochs=10, batch_size=64, rollout_steps=2048,
                 log_std_init=-0.5, save_interval=30_000,
                 grad_clip=0.5, norm_obs=False, norm_rew=False, use_sde=False,
                 sde_sample_freq=4, anneal_lr=False, use_tanh=False,
                 td_lambda_returns=False, seed=None, eval_episodes=10):
        '''Initialize PPO agent'''

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

        # PPO hyperparameters
        self.lr = lr
        self.max_steps = max_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.log_std_init = log_std_init
        self.save_interval = save_interval
        self.grad_clip = grad_clip
        self.norm_obs = norm_obs
        self.norm_rew = norm_rew
        self.rew_norm = RunningRewardNorm(gamma=gamma) if norm_rew else None
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.anneal_lr = anneal_lr
        self.use_tanh = use_tanh
        self.td_lambda_returns = td_lambda_returns

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor-critic network
        self.actor = ActorNetwork(
            state_dim, action_dim, log_std_init=log_std_init,
            norm_obs=norm_obs, use_sde=use_sde, use_tanh=use_tanh
        ).to(self.device)
        self.critic = CriticNetwork(
            state_dim, norm_obs=norm_obs, use_tanh=(use_sde or use_tanh)
        ).to(self.device)
        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Action bounds used to scale the sampled actions
        self.action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)

        # Total environment steps taken, used as the x axis for logging so that
        # curves are comparable across agents that update at different rates
        self.total_steps = 0

        # Logger
        self.env_tag = env_name.lower().replace('-', '_').split('/')[-1]
        self.num_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        self.variant_tag = "ppo"
        hparams = {
            'lr': lr, 'gamma': gamma, 'lam': gae_lambda, 'clip': clip_ratio,
            'ent': entropy_coef, 'bs': batch_size, 'epochs': update_epochs,
            'steps': rollout_steps, 'lsi': log_std_init
        }
        if norm_obs:
            hparams['norm'] = 1
        if norm_rew:
            hparams['nrew'] = 1
        if use_sde:
            hparams['sde'] = 1
        if anneal_lr:
            hparams['anneal'] = 1
        if use_tanh:
            hparams['tanh'] = 1
        if td_lambda_returns:
            hparams['tdlam'] = 1
        self.logger = Logger(self.env_tag, self.env_name, self.variant_tag, self.num_params, hparams=hparams)



    def select_action(self, state):
        '''Select action according to current policy and compute log probability'''
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Sample action (standard Gaussian or state-dependent gSDE)
            raw_action_t, mean, std = self.actor.sample_action(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(raw_action_t).sum(-1)
            entropy = dist.entropy().sum(-1)

            # Get value estimate from critic
            value = self.critic(state_tensor).item()

            # Scale action to environment's action space and clip to valid bounds
            raw_action_1d = raw_action_t.squeeze(0)
            scaled_action = self.action_low + (0.5 * (raw_action_1d + 1.0) * (self.action_high - self.action_low))
            action = np.clip(scaled_action.cpu().numpy().flatten(), self.env.action_space.low, self.env.action_space.high)

            return action, log_prob.cpu().item(), value, entropy.cpu().item(), raw_action_1d.cpu().numpy().flatten()



    def estimate_value(self, state):
        '''Value estimate from the critic for a single state'''
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.critic(state_tensor).item()



    def compute_gae(self, rewards, values, dones, next_value):
        '''Compute Generalized Advantage Estimation'''
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = next_value
        running_advantage = 0.0

        # Compute GAE in reverse order
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            running_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * running_advantage
            advantages[t] = running_advantage

            running_return = rewards[t] + self.gamma * next_non_terminal * running_return
            returns[t] = running_return

        if self.td_lambda_returns:
            returns = advantages + values
        return advantages, returns



    def update_policy(self, states, actions, old_log_probs, advantages, returns, old_values):
        '''Update policy using PPO clipped objective with decoupled actor and critic'''
        if self.anneal_lr:
            frac = max(0.0, 1.0 - self.total_steps / float(self.max_steps))
            lr_now = frac * self.lr
            for pg in self.actor_optimizer.param_groups:
                pg['lr'] = lr_now
            for pg in self.critic_optimizer.param_groups:
                pg['lr'] = lr_now

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)

        # Explained variance diagnostic (before update)
        if len(returns) > 1:
            y_var = torch.var(returns, unbiased=False)
            explained_var = 1.0 - torch.var(returns - old_values, unbiased=False) / (y_var + 1e-8) if y_var > 1e-8 else torch.tensor(0.0, device=self.device)
        else:
            explained_var = torch.tensor(0.0, device=self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        actor_loss_epoch = 0.0
        critic_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        approx_kl_epoch = 0.0
        clipfrac_epoch = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # ACTOR UPDATE
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                curr_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1)

                log_ratio = curr_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_total_loss = actor_loss + self.entropy_coef * entropy_loss
                actor_total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_optimizer.step()
                self.actor.clamp_log_std()

                # CRITIC UPDATE
                curr_values = self.critic(batch_states).squeeze(-1)
                critic_loss = 0.5 * ((curr_values - batch_returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                approx_kl_epoch += approx_kl.item()
                clipfrac_epoch += clipfrac.item()

        num_updates = self.update_epochs * (dataset_size // self.batch_size + int(dataset_size % self.batch_size > 0))
        self._last_diagnostics = {
            'approx_kl': approx_kl_epoch / num_updates,
            'clipfrac': clipfrac_epoch / num_updates,
            'explained_var': float(explained_var.item()),
        }
        return actor_loss_epoch / num_updates, critic_loss_epoch / num_updates, entropy_loss_epoch / num_updates



    def _save_checkpoint(self):
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/{self.logger.run_name}_{self.env_name}_steps_{self.total_steps}.pth'
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'norm_obs': self.norm_obs,
            'use_sde': self.use_sde,
            'use_tanh': self.use_tanh,
            'log_std_init': self.log_std_init,
        }, ckpt_path)
        return ckpt_path

    def train_ppo(self):
        """Train the agent using PPO algorithm with logging"""

        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        actor_loss, value_loss, entropy_loss = 0.0, 0.0, 0.0
        self._last_diagnostics = {'approx_kl': 0.0, 'clipfrac': 0.0, 'explained_var': 0.0}

        i_episode = 0
        last_save = -self.save_interval

        while self.total_steps < self.max_steps:
            reset_seed = self.seed if (self.seed is not None and i_episode == 0) else None
            state, _ = self.env.reset(seed=reset_seed)
            self.actor.reset_noise()
            total_reward = 0.0
            episode_steps = 0

            done = False
            while not done:
                if self.norm_obs:
                    self.actor.update_obs_norm(state)
                    self.critic.update_obs_norm(state)

                action, log_prob, value, entropy, raw_action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                total_reward += reward
                episode_steps += 1

                # Time-limit truncation is not a true terminal state, so bootstrap next value
                stored_reward = self.rew_norm.normalize(reward, done) if self.rew_norm is not None else reward
                if truncated:
                    if self.norm_obs:
                        self.actor.update_obs_norm(next_state)
                        self.critic.update_obs_norm(next_state)
                    stored_reward += self.gamma * self.estimate_value(next_state)

                states.append(state)
                actions.append(raw_action)
                rewards.append(stored_reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(float(done))

                state = next_state
                self.total_steps += 1

                if self.use_sde and self.sde_sample_freq > 0 and (self.total_steps % self.sde_sample_freq == 0):
                    self.actor.reset_noise()

                if len(states) >= self.rollout_steps:
                    next_value = 0.0 if done else self.estimate_value(state)
                    advantages, returns = self.compute_gae(np.array(rewards), np.array(values), np.array(dones), next_value)

                    actor_loss, value_loss, entropy_loss = self.update_policy(
                        states=np.array(states),
                        actions=np.array(actions),
                        old_log_probs=np.array(log_probs),
                        advantages=advantages,
                        returns=returns,
                        old_values=np.array(values)
                    )

                    self.logger.add_value('train/actor_loss', actor_loss, self.total_steps)
                    self.logger.add_value('train/critic_loss', value_loss, self.total_steps)
                    self.logger.add_value('train/entropy_loss', entropy_loss, self.total_steps)
                    self.logger.add_value('train/approx_kl', self._last_diagnostics['approx_kl'], self.total_steps)
                    self.logger.add_value('train/clipfrac', self._last_diagnostics['clipfrac'], self.total_steps)
                    self.logger.add_value('train/explained_variance', self._last_diagnostics['explained_var'], self.total_steps)
                    self.logger.add_value('train/policy_std', torch.exp(self.actor.log_std).mean().item(), self.total_steps)

                    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
                    if self.use_sde:
                        self.actor.reset_noise()

            self.logger.add_value('train/episode_reward', total_reward, self.total_steps)
            self.logger.add_value('train/episode_length', episode_steps, self.total_steps)
            self.logger.add_value('train/episode', i_episode, self.total_steps)

            if self.total_steps - last_save >= self.save_interval:
                last_save = self.total_steps
                self._save_checkpoint()

                avg_eval_reward, frames = utils.evaluate_policy(
                    self.actor, self.env.observation_space, self.env_name,
                    episodes=self.eval_episodes, return_frames=True
                )
                self.logger.add_value('train/eval_mean_reward', avg_eval_reward, self.total_steps)
                self.logger.add_frames('train/eval_video', frames, self.total_steps)

            print(f"Episode {i_episode} - Steps: {self.total_steps}/{self.max_steps} - Total Reward: {total_reward:.2f} - Actor Loss: {actor_loss:.4f} - Critic Loss: {value_loss:.4f}")
            i_episode += 1

        # Save final checkpoint and run final deterministic evaluation
        self._save_checkpoint()
        avg_eval_reward = utils.evaluate_policy(
            self.actor, self.env.observation_space, self.env_name,
            episodes=self.eval_episodes, return_frames=False
        )
        self.logger.add_value('train/eval_mean_reward', avg_eval_reward, self.total_steps)

        print('Training complete')
        self.logger.close()
        self.env.close()

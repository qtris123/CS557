"""
CS 557 RLI Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import gymnasium as gym
import numpy as np
import torch

# Cap the number of frames recorded per video so that the logged videos stay a reasonable size
MAX_VIDEO_FRAMES = 300


def encode_obs(obs, space):
    """
    Convert observations to flat vector representation based on space type.
    
    Args:
        obs: Raw observation from environment
        space: The observation space
        
    Returns:
        Encoded observation as flat numpy array
    """

    if isinstance(space, gym.spaces.Box):
        return np.array(obs, dtype=np.float32).reshape(-1)

    if isinstance(space, gym.spaces.Discrete):
        vec = np.zeros(space.n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
        
    if isinstance(space, gym.spaces.Tuple):
        parts = []
        for sub, val in zip(space.spaces, obs):
            if isinstance(sub, gym.spaces.Discrete):
                one_hot = np.zeros(sub.n, dtype=np.float32)
                one_hot[int(val)] = 1.0
                parts.append(one_hot)
            else:
                raise ValueError('Unsupported subspace in Tuple')
        return np.concatenate(parts).astype(np.float32)
        
    raise ValueError('Unsupported space')


def evaluate_policy(policy, obs_space, env_name, episodes=5, return_frames=False, max_length=None):
    """
    Evaluates the policy in the given environment

    The policy is evaluated deterministically, by using the mean of the action
    distribution rather than a sample from it. Frames are only recorded for the
    first episode, so that all of the episodes can be logged as a single video.
    
    Args:
        policy: The policy to evaluate
        obs_space: Observation space for preprocessing
        env_name: Name of the environment to create
        episodes: Number of episodes to evaluate
        return_frames: Whether to return frames for video logging
        max_length: Maximum length of each episode (None uses the environment's own limit)
        
    Returns:
        mean_reward: Mean reward across episodes
        frames: List of frames if return_frames=True
    """
    env = gym.make(env_name, render_mode="rgb_array" if return_frames else None)

    policy.eval()
    rewards = []
    frames = []
    
    device = next(policy.parameters()).device

    # Scale actions to the environment's action space (for stability)
    low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
    high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    for i_episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        t = 0
        episode_frames = []

        # Only record the first episode
        record = return_frames and i_episode == 0

        if record:
            try:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
                else:
                    episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))
            except Exception as e:
                print(f"Warning: Failed to render initial frame: {e}")
                episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

        while not done:
            # Process observation and select action
            obs_vec = torch.from_numpy(encode_obs(obs, obs_space)).float().unsqueeze(0).to(device)
            
            # Act deterministically. Each policy defines its own deterministic action
            # so that it is evaluated with the same action mapping it trains with.
            with torch.no_grad():
                raw_action = policy.deterministic_action(obs_vec)
                scaled_action = low + (0.5 * (raw_action + 1.0) * (high - low))
            
            # Execute action - ensure proper shape
            action_np = scaled_action.cpu().numpy()
            if len(action_np.shape) > 1 and action_np.shape[0] == 1:
                action_np = action_np.flatten()

            # The policy mean is unbounded, so clip it to the valid action range
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                
            obs, reward, terminated, truncated, _ = env.step(action_np)
            
            # Handle reward if it's an array
            if isinstance(reward, np.ndarray):
                reward = float(reward.item()) if reward.size == 1 else float(reward.sum())

            if record and len(episode_frames) < MAX_VIDEO_FRAMES:
                try:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                    else:
                        if episode_frames:
                            episode_frames.append(episode_frames[-1])
                        else:
                            episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))
                except Exception as e:
                    print(f"Warning: Failed to render frame: {e}")
                    if episode_frames:
                        episode_frames.append(episode_frames[-1])
                    else:
                        episode_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

            episode_reward += reward
            done = terminated or truncated
            t += 1

            if max_length is not None and t >= max_length:
                break
        
        rewards.append(episode_reward)
        if record:
            frames.append(episode_frames)
    
    env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0

    policy.train()

    if return_frames:
        return avg_reward, frames
    
    return avg_reward

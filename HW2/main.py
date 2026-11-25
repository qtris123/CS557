#!/usr/bin/env python3
"""
CS 557 RLI Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import os
import argparse
import torch

# Prevent OpenMP thread oversubscription on high-core-count nodes when unset
if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(1)

from reinforce import REINFORCEAgent
from sac import SACAgent
from ppo import PPOAgent

DATA_DIR = "data"


def main():
    """Entry point for CS 557 HW2."""

    parser = argparse.ArgumentParser(
        description='CS 557 RLI: Homework 2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--reinforce', 
        action="store_true",
        help='Use REINFORCE agent'
    )

    parser.add_argument(
        '--sac', 
        action="store_true",
        help='Use Soft Actor-Critic (SAC) agent'
    )

    parser.add_argument(
        '--ppo',
        action="store_true",
        help='Use Proximal Policy Optimization (PPO) agent'
    )

    parser.add_argument(
        '--env', 
        type=str, 
        default='MountainCarContinuous-v0',
        help='Gym environment (MountainCarContinuous-v0, BipedalWalker-v3)'
    )

    # Common hyperparameters
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Mini-batch size (used by PPO and SAC)'
    )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=1_000_000,
        help='Total number of environment steps to train for'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=30_000,
        help='How frequently (in environment steps) to log progress, videos, and save models'
    )

    parser.add_argument(
        '--log_std_init',
        type=float,
        default=-0.5,
        help='Initial log standard deviation of the Gaussian policy for REINFORCE and PPO '
             '(the parameter is learned from here; SAC predicts its own log std)'
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor for future rewards'
    )

    # SAC specific hyperparameters
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=256,
        help='Hidden layer size for neural networks (matches the 256 used by REINFORCE and PPO)'
    )

    parser.add_argument(
        '--replay_size',
        type=int,
        default=100000,
        help='Size of the replay buffer'
    )

    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Soft update parameter for target network'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Entropy regularization coefficient for SAC'
    )
    
    parser.add_argument(
        '--updates_per_step',
        type=int,
        default=1,
        help='Number of updates per environment step for SAC'
    )
    
    parser.add_argument(
        '--start_steps',
        type=int,
        default=10000,
        help='Number of initial random exploration steps for SAC'
    )

    # PPO specific hyperparameters
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='GAE lambda parameter for PPO'
    )
    
    parser.add_argument(
        '--clip_ratio',
        type=float,
        default=0.2,
        help='PPO clip ratio/epsilon'
    )
    
    parser.add_argument(
        '--value_coef',
        type=float,
        default=0.5,
        help='Value function loss coefficient for PPO'
    )
    
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0.01,
        help='Entropy loss coefficient for PPO'
    )
    
    parser.add_argument(
        '--update_epochs',
        type=int,
        default=10,
        help='Number of epochs to update policy for each batch in PPO'
    )

    parser.add_argument(
        '--rollout_steps',
        type=int,
        default=2048,
        help='Number of environment steps collected per PPO update'
    )

    parser.add_argument(
        '--grad_clip',
        type=float,
        default=0.5,
        help='Gradient norm clipping threshold for PPO (0 disables clipping)'
    )

    parser.add_argument(
        '--norm_obs',
        action='store_true',
        help='Enable running observation normalization in PPO actor and critic'
    )

    parser.add_argument(
        '--norm_rew',
        action='store_true',
        help='Enable running return-std reward scaling in PPO rollout buffer'
    )

    parser.add_argument(
        '--use_sde',
        action='store_true',
        help='Enable Generalized State-Dependent Exploration (gSDE) in PPO'
    )

    parser.add_argument(
        '--sde_sample_freq',
        type=int,
        default=4,
        help='Resample gSDE exploration noise every N steps'
    )

    parser.add_argument(
        '--anneal_lr',
        action='store_true',
        help='Linearly anneal PPO learning rate to 0 across max_steps'
    )

    parser.add_argument(
        '--use_tanh',
        action='store_true',
        help='Use Tanh activations in PPO actor and critic networks'
    )

    parser.add_argument(
        '--td_lambda_returns',
        action='store_true',
        help='Use TD(lambda) returns (advantages + values) for PPO critic target'
    )

    parser.add_argument(
        '--ou_warmup',
        action='store_true',
        help='Use Ornstein-Uhlenbeck temporally correlated noise during SAC start_steps'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--eval_episodes',
        type=int,
        default=10,
        help='Number of deterministic evaluation episodes per checkpoint'
    )


    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.reinforce:
        print("Training REINFORCE agent...")
        reinforce_agent = REINFORCEAgent(
            env_name=args.env,
            lr=args.lr,
            max_steps=args.max_steps,
            gamma=args.gamma,
            log_std_init=args.log_std_init,
            save_interval=args.save_interval,
            seed=args.seed,
            eval_episodes=args.eval_episodes
        )
        reinforce_agent.train_reinforce()

    elif args.sac:
        print("Training SAC agent...")
        sac_agent = SACAgent(
            env_name=args.env,
            lr=args.lr,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            hidden_size=args.hidden_size,
            buffer_size=args.replay_size,
            alpha=args.alpha,
            updates_per_step=args.updates_per_step,
            start_steps=args.start_steps,
            save_interval=args.save_interval,
            ou_warmup=args.ou_warmup,
            seed=args.seed,
            eval_episodes=args.eval_episodes
        )
        sac_agent.train_sac()

    elif args.ppo:
        print("Training PPO agent...")
        ppo_agent = PPOAgent(
            env_name=args.env,
            lr=args.lr,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            update_epochs=args.update_epochs,
            rollout_steps=args.rollout_steps,
            log_std_init=args.log_std_init,
            save_interval=args.save_interval,
            grad_clip=args.grad_clip,
            norm_obs=args.norm_obs,
            norm_rew=args.norm_rew,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            anneal_lr=args.anneal_lr,
            use_tanh=args.use_tanh,
            td_lambda_returns=args.td_lambda_returns,
            seed=args.seed,
            eval_episodes=args.eval_episodes
        )
        ppo_agent.train_ppo()
        
    else:
        print("Please specify an agent to use: --reinforce --ppo --sac")



if __name__ == '__main__':
    main()





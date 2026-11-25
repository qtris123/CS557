"""
CS 557 RLI Homework Assignment 2
Purdue University
Created by: Joseph Campbell and Guven Gergerli
"""

import numpy as np
import os
import time

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, env_tag, env_name, source_tag, num_params, hparams=None):
        """Configure TensorBoard logging."""

        # Include the hyperparameters in the run name so that the runs from a
        # hyperparameter sweep can be told apart in TensorBoard
        hparams = hparams or {}
        run_tag = '_'.join(f'{k}{v}' for k, v in hparams.items())
        base_name = f"{source_tag}_{run_tag}_{int(time.time())}" if run_tag else f"{source_tag}_{int(time.time())}"

        log_dir = os.path.join('runs', env_tag, base_name)
        suffix = 0
        while os.path.exists(log_dir):
            suffix += 1
            log_dir = os.path.join('runs', env_tag, f"{base_name}_{suffix}")

        self.run_name = os.path.basename(log_dir)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Log configuration information
        config = ', '.join(f'{k}={v}' for k, v in hparams.items())
        self.writer.add_text(
            'run/config', 
            f'env={env_name}, source={source_tag}, params={num_params}, {config}'
        )

    def add_value(self, name, y, x):
        """Add a single scalar value to Tensorboard"""

        self.writer.add_scalar(name, y, x)
        self.writer.flush()

    def add_frames(self, name, frames, x, fps=10):
        """Add rendered frames from a list (should be list of lists) of episodes to Tensorboard"""

        video = np.array([np.transpose(frame, [0, 3, 1, 2]) for frame in frames])

        self.writer.add_video('{}'.format(name), video, x, fps=fps)

    def close(self):
        """Flush and close the TensorBoard SummaryWriter."""
        self.writer.flush()
        self.writer.close()

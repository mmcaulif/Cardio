import logging
import time
from collections import deque
from datetime import datetime
from typing import Deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.logging import logging_redirect_tqdm


class Logger:
    def __init__(
        self,
        n_envs=1,
        log_interval=5000,
        episode_window=50,
        tensorboard=False,
        log_dir=None,
        exp_name="exp",
    ) -> None:
        """_summary_

        Args:
            n_envs (int, optional): _description_. Defaults to 1.
            log_interval (int, optional): _description_. Defaults to 5000.
            episode_window (int, optional): _description_. Defaults to 50.
            tensorboard (bool, optional): _description_. Defaults to False.
            log_dir (_type_, optional): _description_. Defaults to None.
            exp_name (str, optional): _description_. Defaults to "exp".

        Raises:
            NotImplementedError: Current Logger class is deprecated. New
                version is not yet ready.
        """
        raise NotImplementedError
        self.n_envs = n_envs
        self.log_interval = log_interval
        self.tensorboard = tensorboard

        if self.tensorboard:
            dir = ""

            if log_dir:
                dir += log_dir + "/"

            date_key = datetime.now().strftime("%Y-%m-%d")
            time_key = datetime.now().strftime("%H-%M-%S")

            dir += exp_name + "_" + date_key + "_" + time_key + "/"
            self.writer = SummaryWriter(dir)

        self.timestep = 0
        self.episodes = 0
        self.initial_time = time.time()
        self.prev_time = 0

        self.running_reward = 0
        self.episodic_rewards: Deque = deque(maxlen=episode_window)

        logging.basicConfig(
            format="%(asctime)s: %(message)s",
            datefmt=" %I:%M:%S %p",
            level=logging.INFO,
            # handlers=[RichHandler()]
        )

        self.logger = logging.getLogger()

    def step(self, reward, done):
        self.timestep += 1
        self.running_reward += reward

        if done:
            self.episodes += 1
            self.episodic_rewards.append(self.running_reward)
            self.running_reward = 0

        if self.timestep % self.log_interval == 0:
            total_time = time.time() - self.initial_time
            d_time = total_time - self.prev_time
            fps = self.log_interval / d_time

            metrics = {
                "Timesteps": self.timestep,
                "Episodes": self.episodes,
                "Episodic reward": np.round(np.mean(self.episodic_rewards), 4),
                "Time passed": round(total_time, 2),
                "Env steps per second": int(fps),
            }
            self.prev_time = total_time

            with logging_redirect_tqdm():
                self.logger.info(metrics)

            if self.tensorboard:
                # Figure out how to write metrics dictionary
                self.writer.add_scalar(
                    "rollout/ep_rew_mean", np.mean(self.episodic_rewards), self.timestep
                )

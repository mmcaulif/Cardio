from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import logging
import numpy as np
from datetime import datetime

"""
-Maybe change to Rich Logger
-implement tensorboard next
-make logger compatible with VectorCollector next

# https://docs.python.org/3/howto/logging.html
# https://www.tensorflow.org/tensorboard
# https://pytorch.org/docs/stable/tensorboard.html
"""

class Logger():
    def __init__(
            self,
            n_envs = 1,
            log_interval=2000,
            episode_window=20,
            tensorboard=False,
            log_dir=None,
            exp_name='exp'
        ) -> None:
        
        logging.basicConfig(format='%(asctime)s: %(message)s', datefmt=' %I:%M:%S %p', level=logging.INFO)
        self.n_envs = n_envs
        self.log_interval = log_interval
        # self.episode_window = episode_window
        self.tensorboard = tensorboard

        if self.tensorboard:
            dir = ''

            if log_dir:
                dir += log_dir + '/'

            # time_key = str(int(time.time()//1))
            
            # Changed to the below to be more in line with Hydra
            date_key = datetime.now().strftime('%Y-%m-%d')
            time_key = datetime.now().strftime('%H-%M-%S')

            dir += exp_name + '_' + date_key + '_' + time_key + '/'
            self.writer = SummaryWriter(dir)
        
        self.timestep = 0
        self.episodes = 0
        if n_envs == 1:
            self.running_reward = 0
        else:
            self.running_reward = np.zeros(n_envs)

        self.episodic_rewards = deque(maxlen=episode_window)
        
    def step(self, reward, done, truncated):
        self.timestep += 1
        self.running_reward += reward

        if done or truncated:
            self.episodes += 1
            self.episodic_rewards.append(self.running_reward)
            self.running_reward = 0

        if self.timestep % self.log_interval == 0:
            logging.info(f'Timesteps: {self.timestep}, Episodes: {self.episodes}, Avg. reward is {np.mean(self.episodic_rewards)}')

            if self.tensorboard:
                self.writer.add_scalar('rollout/ep_rew_mean', np.mean(self.episodic_rewards), self.timestep)

    def vector_step(self, rewards, dones, truns):
        self.timestep += self.n_envs

        self.running_reward += rewards

        for i, (done, truncated) in enumerate(zip(dones, truns)):
            if done or truncated:
                self.episodes += 1
                self.episodic_rewards.append(self.running_reward[i])
                self.running_reward[i] = 0

        if (self.timestep//self.n_envs) % self.log_interval == 0:
            logging.info(f'Timesteps: {self.timestep}, Episodes: {self.episodes}, Avg. reward is {np.mean(self.episodic_rewards)}')


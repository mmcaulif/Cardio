from collections import deque
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np

"""
-Maybe change to Rich Logger
-implement tensorboard next
-make logger compatible with VectorCollector next

# https://docs.python.org/3/howto/logging.html
# https://www.tensorflow.org/tensorboard
"""

class Logger():
    def __init__(
            self,
            log_interval,
            episode_window,
            tensorboard,
            exp_name
        ) -> None:
        
        logging.basicConfig(format='%(asctime)s: %(message)s', datefmt=' %I:%M:%S %p', level=logging.INFO)

        self.log_interval = log_interval
        # self.episode_window = episode_window
        self.tensorboard = tensorboard

        if self.tensorboard:
            dir = 'run/'
            if exp_name:
                dir += exp_name + '/'

            self.writer = SummaryWriter(dir)
        
        self.timestep = 0
        self.episodes = 0
        self.running_reward = 0
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
                self.writer.add_scalar('Avg. reward', np.mean(self.episodic_rewards), self.timestep)






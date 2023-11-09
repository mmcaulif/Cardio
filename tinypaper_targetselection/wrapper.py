import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TransformObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(
            0, 1, shape=(env.game.state_shape()[2],10,10), dtype=np.uint8
        )

        self.f = lambda s: np.transpose(s, (2, 0, 1))

    def observation(self, observation):        
        return self.f(observation)
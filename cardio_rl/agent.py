from gymnasium import Env
import numpy as np


class Agent:
    def __init__(self):
        pass

    def _init_env(self, env: Env):
        self.env = env

    def setup(self, env):
        self._init_env(env)

    def view(self, transition: callable, extra: dict):
        return extra

    def step(self, state: np.ndarray):
        return self.env.action_space.sample(), {}

    def update(self, data: dict):
        pass

    def terminal(self):
        pass

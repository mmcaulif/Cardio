from gymnasium import Env
import numpy as np


class Agent:
    def __init__(self, env: Env):
        self.env = env

    def view(self, transition: dict[str, np.array], extra: dict):
        return extra

    def step(self, state: np.ndarray):
        return self.env.action_space.sample(), {}
    
    def eval_step(self, state: np.ndarray):
        return self.step(state)

    def update(self, data: list[dict[str, np.array]]):
        pass

    def terminal(self):
        pass


from gymnasium import Env

class Agent:
    def __init__(self):
        pass

    def _init_env(self, env: Env):
        self.env = env

    def setup(self, env):
        self._init_env(env)

    def step(self, state):
        return self.env.action_space.sample(), {}
    
    def update(self, data):
        pass

    def terminal(self):
        pass
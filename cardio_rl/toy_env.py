import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ToyEnv(gym.Env):
    def __init__(self, maxlen=10, discrete=True) -> None:
        self.maxlen = maxlen
        self.discrete = discrete
        self.count = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, self.maxlen, shape=[5,], dtype=np.float32)

    def step(self, action):
        self.count += 1
        state = np.ones(5) * self.count
        state[-1] = action
        if self.count == self.maxlen:
            return np.array(state), 1, True, False, {}

        return np.array(state), 0.1*self.count, False, False, {}

    def reset(self):
        self.count = 0
        return np.ones(5) * self.count, {}


def main():
    env = ToyEnv()

    s_t = env.reset()
    while True:
        print(s_t)
        a_t = env.action_space.sample()
        s_tp1, r, d, t, i = env.step(a_t)
        print(s_t, a_t, s_tp1, r, d)
        s_t = s_tp1
        if d:
            break


if __name__ == "__main__":
    main()

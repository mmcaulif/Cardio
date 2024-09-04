import gymnasium as gym
import numpy as np
from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3.common import atari_wrappers


class AtariWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, eval: bool = False):
        if not eval:
            env = atari_wrappers.AtariWrapper(env)
        else:
            env = atari_wrappers.AtariWrapper(
                env, terminal_on_life_loss=False, clip_reward=False
            )

        env = FrameStack(env, num_stack=4)
        super().__init__(env)

    def step(self, a):
        s, r, d, t, info = super().step(a)
        return np.array(s) / 255.0, r, d, t, info

    def reset(self):
        s, info = self.env.reset()
        return np.array(s) / 255.0, info

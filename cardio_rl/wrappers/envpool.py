"""Atari preprocessing for individual envpool environments."""

#  import gymnasium as gym
# import numpy as np


# class EnvPoolWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env):
#         """Wrapper that allows Envpool atari environments to be used as a drop
#         in replacement for gymnasium Atari environments by ensuring action
#         dtype is int32 and squeezing the MDP elements.

#         Args:     env (gym.Env): _description_

#         Raises:     NotImplementedError: _description_
#         """
#         raise NotImplementedError
#         super().__init__(env)

#     def step(self, a):
#         if not isinstance(a, np.int32):
#             a = np.astype(a, np.int32)

#         # if len(a.shape) < 2:
#         #     a = np.expand_dims(a, 0)

#         s, r, d, t, info = super().step(a)
#         return s[0] / 255.0, r[0], d[0], t[0], info

#     def reset(self):
#         s, info = self.env.reset()
#         return s[0] / 255.0, info

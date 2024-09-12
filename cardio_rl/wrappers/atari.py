import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3.common import atari_wrappers


class AtariWrapper(gym.Wrapper):
    """This needs to be triple checked..."""

    def __init__(
        self, env: gym.Env, action_repeat_probability: float = 0.0, eval: bool = False
    ):
        if not eval:
            env = atari_wrappers.AtariWrapper(
                env, action_repeat_probability=action_repeat_probability
            )
        else:
            env = atari_wrappers.AtariWrapper(
                env,
                terminal_on_life_loss=False,
                clip_reward=False,
                action_repeat_probability=action_repeat_probability,
            )

        env = FrameStack(env, num_stack=4)
        super().__init__(env)

        self.observation_space: Box = Box(low=0.0, high=1.0, shape=(84, 84, 4))

    def step(self, a):
        s, r, d, t, info = super().step(a)
        return self._from_lazyframes(s), r, d, t, info

    def reset(self, seed=None, options=None):
        del seed
        del options
        s, info = self.env.reset()
        return self._from_lazyframes(s), info

    def _from_lazyframes(self, x):
        return np.array(x).squeeze(-1).transpose(1, 2, 0) / 255.0

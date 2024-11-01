"""ALE environments wrapper without stable_baselines3 dependency."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack, LazyFrames
from gymnasium.wrappers.transform_reward import TransformReward


class AtariGymnasiumWrapper(gym.Wrapper):
    """Wrapper for Atari preprocessing."""

    def __init__(
        self, env: gym.Env, action_repeat_probability: float = 0.0, eval: bool = False
    ):
        """Wrap environment with the traditional Atari wrappers.

        Uses the build in gymnasium AtariPreprocessing wrapper with
        framestacks, reward clipping and any necessary observation
        transformations (such as moving channels to the final dimension
        and normalising values) to a given gymnasium ALE environment.

        TODO: needs to be tested to ensure the functionality is the
        same as the one using stable_baselines3.

        Args:
            env (gym.Env): The instantiated gymnasium environment.
            action_repeat_probability (float, optional): Probaility of
                a sticky action. Defaults to 0.0.
            eval (bool, optional): If set to true, rewards are not
                clipped and episodes do not end on life loss. Defaults
                to False.
        """
        if not eval:
            env = AtariPreprocessing(env, scale_obs=True)
            env = TransformReward(env, np.sign)
        else:
            env = AtariPreprocessing(env, terminal_on_life_loss=False, scale_obs=True)

        env = FrameStack(env, num_stack=4)
        self.action_repeat_probability = action_repeat_probability
        super().__init__(env)

        self.observation_space: Box = Box(low=0.0, high=1.0, shape=(84, 84, 4))

    def step(self, a: np.ndarray):
        """Step through the environment.

        Using thr given action, check if a sticky action should be
        performed and then step through the environment.

        Args:
            a (np.ndarray): The action taken by the agent.

        Returns:
            tuple: The next state, reward, terminal, truncation and
                info values.
        """
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = a
        s, r, d, t, info = super().step(self._sticky_action)
        return self._from_lazyframes(s), r, d, t, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment.

        Reset the environment to an initial state given a seed. If no
        seed is given, one is generated randomly within the
        environment.

        Args:
            seed (int | None, optional): The seed for the reset.
                Defaults to None.
            options (dict, optional): Additional information to
                specify how the environment is reset. Defaults to None.

        Returns:
            tuple: The initial state and info dict.
        """
        self._sticky_action = np.array(0)
        s, info = self.env.reset(seed=seed, options=options)
        return self._from_lazyframes(s), info

    def _from_lazyframes(self, x: LazyFrames) -> np.ndarray:
        """Process a state from laxy frames to numpy array.

        Convert a lazyframe to an observation with the correct
        dimensions and scale by transnposing the input.

        Args:
            x (LazyFrames): LazyFrames to convert to observation.

        Returns:
            np.ndarray: The processed observation for the agent.
        """
        return np.array(x).transpose(1, 2, 0)

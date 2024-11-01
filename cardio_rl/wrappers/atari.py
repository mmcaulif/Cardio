"""ALE environments wrapper."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers.frame_stack import FrameStack, LazyFrames
from stable_baselines3.common import atari_wrappers


class AtariWrapper(gym.Wrapper):
    """Wrapper for Atari preprocessing."""

    def __init__(
        self, env: gym.Env, action_repeat_probability: float = 0.0, eval: bool = False
    ):
        """Wrap environment with the traditional Atari wrappers.

        Uses the stable_baselines3 atari_wrapper with framestacks,
        and any necessary observation transformations (such as moving
        channels to the final dimension and normalising values) to a
        given gymnasium ALE environment.

        Args:
            env (gym.Env): The instantiated gymnasium environment.
            action_repeat_probability (float, optional): Probaility of
                a sticky action. Defaults to 0.0.
            eval (bool, optional): If set to true, rewards are not
                clipped and episodes do not end on life loss. Defaults
                to False.
        """
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

    def step(self, a: np.ndarray):
        """Step through the environment.

        Using the given action, check if a sticky action should be
        performed and then step through the environment. Sticky actions
        are handled by stable_baseline3's AtariWrapper.

        Args:
            a (np.ndarray): The action taken by the agent.

        Returns:
            tuple: The next state, reward, terminal, truncation and
                info values.
        """
        s, r, d, t, info = super().step(a)
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
        s, info = self.env.reset(seed=seed, options=options)
        return self._from_lazyframes(s), info

    def _from_lazyframes(self, x: LazyFrames) -> np.ndarray:
        """Process a state from laxy frames to numpy array.

        Convert a lazyframe to an observation with the correct
        dimensions and scale by squeezing and transposing the input,
        and then dividing by 255.

        Args:
            x (LazyFrames): LazyFrames to convert to observation.

        Returns:
            np.ndarray: The processed observation for the agent.
        """
        return np.array(x).squeeze(-1).transpose(1, 2, 0) / 255.0

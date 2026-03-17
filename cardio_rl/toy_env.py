"""Toy Environment for debugging."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ToyEnv(gym.Env):
    """Toy environment for debugging."""

    def __init__(self, maxlen: int = 10, discrete: bool = True) -> None:
        """Intilise the Environment.

        A toy gym Environment useful for debugging Cardio's components.
        Given a maximum environment length, the environment returns an
        observation of [t, t, t, t, a] where t is the total number of
        steps and a is the last action given to the environment. Reward
        at each timesteps is t * 0.1. The Environment is terminal once
        maxlen number of steps have neen taken. The environment has a
        discrete action space of 2 actions by default.

        Args:
            maxlen (int, optional): Length of episodes. Defaults to 10.
            discrete (bool, optional): Sets the action space to
                discrete. Defaults to True.

        Raises:
            NotImplementedError: If self.discrete is set to False.
        """
        self.maxlen = maxlen
        self.discrete = discrete
        self.t = 0

        if discrete:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(
            0,
            self.maxlen,
            shape=[
                5,
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        """Step through the environment.

        Using the given action, step through the environment and return
        the obervation that includes that action.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            tuple: The next state, reward, terminal, truncation and
                info values.
        """
        self.t += 1
        state = np.ones(5) * self.t
        state[-1] = action if self.discrete else action[0]
        if self.t == self.maxlen:
            return np.array(state), 1, True, False, {}

        return np.array(state), 0.1 * self.t, False, False, {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment.

        Reset the environment to an initial state. Ignores the seed
        and options as environment is deterministic.

        Args:
            seed (int | None, optional): The seed for the reset.
                Defaults to None.
            options (dict, optional): Additional information to
                specify how the environment is reset. Defaults to None.

        Returns:
            tuple: The initial state and info dict.
        """
        del seed
        del options
        self.t = 0
        return np.ones(5) * self.t, {}

from typing import Any

import numpy as np

from cardio_rl.types import Environment, Transition


class Agent:
    """Base class for Cardio agents. Provides base methods that can be used as
    a dummy agent or inherited and overriden when implementing your own agent.

    Attributes:
        env (Environment): The gym environment used to collect transitions and
            train the agent.
    """

    def __init__(self, env: Environment):
        """_summary_

        Args:
            env (Environment): The gym environment used to collect transitions and
                train the agent.
        """
        self.env = env

    def view(self, transition: Transition, extra: dict) -> dict:
        """_summary_

        Args:
            transition (Transition): _description_
            extra (dict): _description_

        Returns:
            dict: _description_
        """
        del transition
        return extra

    def step(self, state: np.ndarray) -> tuple[Any, dict]:
        """_summary_

        Args:
            state (np.ndarray): _description_

        Returns:
            tuple[Any, dict]: _description_
        """
        del state
        return self.env.action_space.sample(), {}

    def eval_step(self, state: np.ndarray) -> Any:
        """_summary_

        Args:
            state (np.ndarray): _description_

        Returns:
            Any: _description_
        """
        a, _ = self.step(state)
        return a

    def update(self, data: list[Transition]) -> dict:
        """_summary_

        Args:
            data (list[Transition]): _description_

        Returns:
            dict: _description_
        """
        del data
        return {}

    def terminal(self):
        """_summary_"""
        pass

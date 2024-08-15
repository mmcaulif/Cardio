import numpy as np
from gymnasium import Env

from cardio_rl.types import Transition


class Agent:
    """_summary_"""

    def __init__(self, env: Env):
        """_summary_

        Args:
            env (Env): _description_
        """
        self.env = env

    def view(self, transition: Transition, extra: dict):
        """_summary_

        Args:
            transition (Transition): _description_
            extra (dict): _description_

        Returns:
            _type_: _description_
        """
        return extra

    def step(self, state: np.ndarray):
        """_summary_

        Args:
            state (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        return self.env.action_space.sample(), {}

    def eval_step(self, state: np.ndarray):
        """_summary_

        Args:
            state (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        a, _ = self.step(state)
        return a

    def update(self, data: list[Transition]):
        """_summary_

        Args:
            data (list[Transition]): _description_
        """
        pass

    def terminal(self):
        """_summary_"""
        pass

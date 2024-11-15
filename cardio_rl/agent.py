"""Contains the Cardio base agent class."""

from typing import Any

import numpy as np

from cardio_rl.types import Environment, Transition


class Agent:
    """Base class for Cardio agents.

    Provides base methods that can be used as a dummy agent or inherited
    and overriden when implementing your own agent.
    """

    def __init__(self, env: Environment):
        """Intiaslises the Agent class.

        Allows the agent to perform any initialisation necessary such as for
        neural networks or setting hyperparameters.

        Args:
        env (Environment): The gym environment used to collect transitions and
            train the agent.
        """
        self.env = env

    def view(self, transition: Transition, extra: dict) -> dict:
        """Expose a full transition and any extras back to the agent.

        Allows the agent to modify the extra elements. Called by the gatherer
        each environment step.

        Args:
            transition (Transition): Most recent transition.
            extra (dict): Most recent extras.

        Returns:
            dict: Updated extras.
        """
        del transition
        return extra

    def step(self, state: np.ndarray) -> tuple[Any, dict]:
        """Given a state, produce an action and any additional extras.

        Called during training. By default produces a random action and
        an empty extras dict.

        Args:
            state (np.ndarray): The observation the agent sees.

        Returns:
        tuple[Any, dict]: The action taken for the given
            state and extras to store.
        """
        del state
        return self.env.action_space.sample(), {}

    def eval_step(self, state: np.ndarray) -> Any:
        """Given a state, produce an action.

        Called during evaluation. By default calls the step method.

        Args:
            state (np.ndarray): The observation the agent sees.

        Returns:
            Any: The action taken for the given state.
        """
        a, _ = self.step(state)
        return a

    def update(self, data: Transition | list[Transition]) -> dict:
        """Perform any updates given a batch of transitions.

        When using the off-policy runner, the agent can return values
        and indices to be overriden in the buffer.

        Args: data (Transition | list[Transition]): Transition data used
            by the agent to update any internal decision making.

        Returns: dict: Indices and corresponding values to override.
            Must return the indices as elements under the "idxs" key.
        """
        del data
        return {}

    def terminal(self):
        """Called at the end of every episode."""
        pass

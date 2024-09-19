import itertools

import numpy as np

from cardio_rl.agent import Agent
from cardio_rl.gatherers.gatherer import Gatherer
from cardio_rl.types import Transition


class VectorGatherer(Gatherer):
    """A modified gatherer that is used for parallel vectorised environments.
    The step buffer is unused and instead enviornment transitions are passed
    directly into the transition buffer. The step method is the only modified
    method from the parent class and its mainly removing code related to
    terminal episodes and step-to-transition processing for n-step transitions.
    The VectorGatherer is currently incompatible with n-step transitions and it
    is likely to stay this way as they are not frequently used in on-policy
    strategies (and introduce a lot of engineering difficulties).

    Attributes:
        n_step (int, optional): Number of environment steps to store
            per-transition. Defaults to 1.
        transition_buffer (deque): Double ended queue used to store processed transitions. Used directly in the VectorGatherer.
        step_buffer (deque): Double ended queue used to store individual environment transitions. Skipped in the VectorGatherer.
        state (np.ndarray): The current state of the environment.
    """

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> list[Transition]:
        """A simplified version of the defautl gatherer's step method that
        removes the use of the step buffer and accounts for vectorised
        environments auto-reset functionality.

        Args:
            agent (Agent): The agent to step through environment with.
            length (int): The length of the rollout. If set to -1
                it performs one episode.

        Returns:
            list[Transition]: The contents of the transition buffer
                as a list.
        """
        iterable = range(length) if length > 0 else itertools.count()
        for _ in iterable:
            a, ext = agent.step(self.state)
            next_state, r, d, t, _ = self.env.step(a)
            done = np.logical_or(d, t)

            transition = {"s": self.state, "a": a, "r": r, "s_p": next_state, "d": done}
            ext = agent.view(transition, ext)
            transition.update(ext)
            self.transition_buffer.append(transition)

            self.state = next_state
            if any(done):
                # Not currently used for anything...
                idxs = np.where(done == 1.0)
                del idxs

        # Process the transition buffer
        transitions = list(self.transition_buffer)
        self.transition_buffer.clear()
        return transitions

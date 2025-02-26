"""Vectorised Environment Transition gatherer."""

import itertools

import numpy as np

from cardio_rl.agent import Agent
from cardio_rl.gatherers.gatherer import Gatherer
from cardio_rl.types import Transition


class VectorGatherer(Gatherer):
    """A modified gatherer for vectorised environments."""

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> tuple[list[Transition], list[float], list[int], int]:
        """Step through the environments with an agent.

        A simplified version of the default gatherer's step method that
        removes the use of the step buffer and accounts for vectorised
        environments auto-reset functionality. The VectorGatherer is
        currently incompatible with n-step transitions and it is likely
        to stay this way as they are not frequently used in on-policy
        strategies (and introduce a lot of engineering difficulties).

        Args:
            agent (Agent): The agent to step through environment with.
            length (int): The length of the rollout. If set to -1 it
                performs one episode.

        Returns:
            list[Transition]: The contents of the transition buffer as
                a list.
        """
        iterable = iter(range(length)) if length > 0 else itertools.count()
        for _ in iterable:
            self.t += self.n_envs
            a, ext = agent.step(self.state)
            next_state, r, term, trun, _ = self.env.step(a)
            done = np.logical_or(term, trun)

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
        return transitions, [], [], 0

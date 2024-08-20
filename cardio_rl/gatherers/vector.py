import itertools
from collections import deque
from typing import Deque

import numpy as np

from cardio_rl.agent import Agent
from cardio_rl.gatherers.gatherer import Gatherer
from cardio_rl.types import Transition


class VectorGatherer(Gatherer):
    def __init__(
        self,
    ) -> None:
        self.transition_buffer: Deque = deque()

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> list[Transition]:
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
                idxs = np.where(done == 1.0)
                del idxs

        # Process the transition buffer
        transitions = list(self.transition_buffer)
        self.transition_buffer.clear()
        return transitions

    def reset(self) -> None:
        self.transition_buffer.clear()
        self.env.reset()

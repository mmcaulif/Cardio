import itertools
from collections import deque
from typing import Deque

import numpy as np
from gymnasium import Env

from cardio_rl.agent import Agent
from cardio_rl.types import Transition


class VectorGatherer:
    def __init__(
        self,
        n_step: int = 1,
    ) -> None:
        self.n_step = n_step

        assert (
            self.n_step == 1
        ), "n-step greater than 1 not implemented for vector environments yet'"

        self.transition_buffer: Deque = deque()
        self.step_buffer: Deque = deque(maxlen=n_step)

    def init_env(self, env: Env):
        self.env = env
        self.state, _ = self.env.reset()

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

            self.step_buffer.append(transition)

            if len(self.step_buffer) == self.n_step:
                step = {
                    "s": self.step_buffer[0]["s"],
                    "a": self.step_buffer[0]["a"],
                    "r": self.step_buffer[0]["r"]
                    if self.n_step == 1
                    else np.array([step["r"] for step in self.step_buffer]),
                    "s_p": self.step_buffer[-1]["s_p"],
                    "d": self.step_buffer[-1]["d"],
                }

                self.transition_buffer.append(step)

            self.state = next_state
            if any(done):
                idxs = np.where(done == 1.0)
                del idxs

                # for i, val in enumerate(done):
                #     if self.n_step > 1:
                #         self._flush_step_buffer()
                #
                #     self.step_buffer.clear()
                #
                # agent.terminal()

        # Process the transition buffer
        transitions = list(self.transition_buffer)
        self.transition_buffer.clear()
        return transitions

    def reset(self) -> None:
        self.step_buffer.clear()
        self.transition_buffer.clear()
        self.env.reset()

    def _flush_step_buffer(self) -> None:
        """When using n-step transitions and reaching a terminal
        state, use the remaining individual steps in the step_buffer
        to not waste information i.e. iterate through states and
        pad reward. Ignore first step as that has already been
        added to transition buffer.
        """

        remainder = len(self.step_buffer)

        if remainder < self.n_step:
            start = 0
        else:
            start = 1

        for i in range(start, remainder):
            temp = list(self.step_buffer)[i:]
            pad = [0.0] * i
            step = {
                "s": temp[0]["s"],
                "a": temp[0]["a"],
                "r": np.array([step["r"] for step in temp] + pad),
                "s_p": temp[-1]["s_p"],
                "d": temp[-1]["d"],
            }

            self.transition_buffer.append(step)

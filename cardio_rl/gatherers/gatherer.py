import itertools
from collections import deque
from typing import Deque, Optional

import numpy as np
from gymnasium import Env
from numpy.typing import NDArray

from cardio_rl.agent import Agent
from cardio_rl.logger import Logger
from cardio_rl.types import Transition


class Gatherer:
    def __init__(
        self,
        n_step: int = 1,
        logger_kwargs: Optional[dict] = None,
    ) -> None:
        self.n_step = n_step

        if logger_kwargs is None:
            logger_kwargs = {}

        self.logger = Logger(**logger_kwargs)
        self.transition_buffer: Deque = deque()
        self.step_buffer: Deque = deque(maxlen=n_step)

    def _init_env(self, env: Env):
        self.env = env
        self.state, _ = self.env.reset()

    def _env_step(self, agent: Agent, s: NDArray):
        a, ext = agent.step(s)
        s_p, r, d, t, _ = self.env.step(a)
        self.logger.step(r, d, t)
        d = d or t

        transition = {"s": s, "a": a, "r": r, "s_p": s_p, "d": d}
        ext = agent.view(transition, ext)
        transition.update(ext)

        return transition, s_p, d, t

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> list[Transition]:
        iterable = range(length) if length > 0 else itertools.count()
        for _ in iterable:
            transition, next_state, done, trun = self._env_step(agent, self.state)
            self.step_buffer.append(transition)

            if len(self.step_buffer) == self.n_step:
                step = {
                    "s": self.step_buffer[0]["s"],
                    "a": self.step_buffer[0]["a"],
                    "r": np.array([step["r"] for step in self.step_buffer]),
                    "s_p": self.step_buffer[-1]["s_p"],
                    "d": self.step_buffer[-1]["d"],
                }
                self.transition_buffer.append(step)

            self.state = next_state
            if done or trun:
                if self.n_step > 1:
                    self._flush_step_buffer()
                self.state, _ = self.env.reset()
                self.step_buffer.clear()
                agent.terminal()
                # For evaluation and/or reinforce
                if length == -1:
                    break

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

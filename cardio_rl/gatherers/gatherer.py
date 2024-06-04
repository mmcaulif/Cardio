from collections import deque
import itertools
from typing import Deque, Optional
from gymnasium import Env
import numpy as np
from cardio_rl.logger import Logger
from cardio_rl.agent import Agent
from cardio_rl import Transition


class Gatherer:
    def __init__(
        self,
        n_step: int = 1,
        take_every: int = 1,
        logger_kwargs: Optional[dict] = None,
    ) -> None:
        self.n_step = n_step
        self.take_every = take_every
        self.take_count = 0

        if logger_kwargs is None:
            logger_kwargs = {}

        self.logger = Logger(**logger_kwargs)
        self.step_buffer: Deque = deque(maxlen=n_step)

    def _init_env(self, env: Env):
        self.env = env
        self.state, _ = self.env.reset()

    def _env_step(self, agent: Agent, s: np.array):
        a, ext = agent.step(s)
        s_p, r, d, t, _ = self.env.step(a)
        self.logger.step(r, d, t)
        d = d or t

        """
        In the interest of making it easier in cardio to track and pass different features and values
        between components it could be good to move towards a dictionary/dataclass approach for 
        timesteps. Using pytree utils this could be relatively straight forward and extensible.

        update: this is now partially implemented but needs to be fully extended to allow for n-step etc.
        """

        transition = {"s": s, "a": a, "r": r, "s_p": s_p, "d": d}
        ext = agent.view(transition, ext)
        transition.update(ext)

        return transition, s_p, d, t

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> list[Transition]:
        # For eval or for reinforce
        if length == -1:
            ret_if_term = True
            iterator = itertools.count()
        else:
            ret_if_term = False
            iterator = range(length)  # type: ignore

        gather_buffer: Deque = deque(maxlen=length)

        for _ in iterator:
            transition, next_state, done, trun = self._env_step(agent, self.state)
            self.step_buffer.append(transition)
            if len(self.step_buffer) == self.n_step:
                    
                if self.n_step == 1:
                    # No idea why but if not converted to a list this overrides the whole gather buffer
                    gather_buffer.append(*list(self.step_buffer))
                else:
                    n_step = {
                        "s": self.step_buffer[0]['s'], 
                        "a": self.step_buffer[0]['a'], 
                        "r": [step['r'] for step in self.step_buffer],
                        "s_p": self.step_buffer[-1]['s_p'],
                        "d": self.step_buffer[-1]['d'],
                    }
                    gather_buffer.append(n_step)

                self.take_count += 1

            self.state = next_state
            if done or trun:
                self.state, _ = self.env.reset()
                self.step_buffer = deque(maxlen=self.n_step)
                agent.terminal()
                if ret_if_term:
                    return list(gather_buffer)

        return list(gather_buffer)

    def reset(self) -> None:
        self.step_buffer.clear()
        self.env.reset()

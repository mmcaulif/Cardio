from collections import deque
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

        if logger_kwargs is None:
            logger_kwargs = {}

        self.logger = Logger(**logger_kwargs)
        self.gather_buffer: Deque = deque()
        self.step_buffer: Deque = deque(maxlen=n_step)

    def _init_env(self, env: Env):
        self.env = env
        self.state, _ = self.env.reset()

    def _env_step(self, agent: Agent, s: np.array):
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

        steps_added = 0
        while steps_added < length and length > 0:
            transition, next_state, done, trun = self._env_step(agent, self.state)
            self.step_buffer.append(transition)
            
            if len(self.step_buffer) == self.n_step:
                step = {
                    "s": self.step_buffer[0]['s'], 
                    "a": self.step_buffer[0]['a'], 
                    "r": np.array([step['r'] for step in self.step_buffer]),
                    "s_p": self.step_buffer[-1]['s_p'],
                    "d": self.step_buffer[-1]['d'],
                }
                self.gather_buffer.append(step)
                steps_added += 1

            self.state = next_state
            if done or trun:
                self.state, _ = self.env.reset()
                self.step_buffer = deque(maxlen=self.n_step)
                agent.terminal()
                # For eval or for reinforce
                if length == -1:
                    break
        
        # Process the gather buffer
        output_batch = list(self.gather_buffer)
        self.gather_buffer.clear()
        return output_batch

    def reset(self) -> None:
        self.step_buffer.clear()
        self.gather_buffer.clear()
        self.env.reset()

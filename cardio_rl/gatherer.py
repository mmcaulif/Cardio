from collections import deque
from typing import Optional
from gymnasium import Env
from cardio_rl.logger import Logger
from cardio_rl.agent import Agent

class Gatherer:
    def __init__(
        self,
        warmup_len: int = 1_000,
        n_step: int = 1,
        take_every: int = 1,
        logger_kwargs: Optional[dict] = None
    ) -> None: 

        self.warmup_len = warmup_len
        self.n_step = n_step   
        self.take_every = take_every 
        self.take_count = 0

        if logger_kwargs == None:
            logger_kwargs = {}

        self.logger = Logger(**logger_kwargs)
        self.step_buffer = deque(maxlen=self.n_step)
    
    def _init_env(self, env: Env):
        self.env = env
        self.state, _ = self.env.reset()

    def _env_step(self, agent: Agent, s):
        a, ext = agent.step(s)
        s_p, r, d, t, _ = self.env.step(a)
        self.logger.step(r, d, t)
        d = d or t

        """
        In the interest of making it easier in cardio to track and pass different features and values
        between components it could be good to move towards a dictionary/dataclass approach for 
        timesteps. Using pytree utils this could be relatively straightforward and extensible.

        transition = {
            's': s, 
            'a': a, 
            'r': r, 
            's_p': s_p,
            'd': d}
        transition.update(ext)
        """
        
        transition = (s, a, r, s_p, d)

        return transition, s_p, d, t

    def step(
        self,
        agent: Agent,
        length: Optional[int] = None,
        ret_if_term: bool = False
    ):
        gather_buffer = deque()

        for _ in range(length):
            transition, next_state, done, trun = self._env_step(agent, self.state)
            self.step_buffer.append(transition)
            if len(self.step_buffer) == self.n_step:
                
                if self.take_count % self.take_every == 0:
                    if self.n_step == 1:
                        gather_buffer.append(*list(self.step_buffer))
                    else:
                        gather_buffer.append(list(self.step_buffer))
 
                self.take_count += 1

            self.state = next_state
            if done or trun:
                self.state, _ = self.env.reset()
                self.step_buffer = deque(maxlen=self.n_step)
                agent.terminal()
                if ret_if_term:
                    return list(gather_buffer)

        return list(gather_buffer)
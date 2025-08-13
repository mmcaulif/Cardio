"""Main Cardio Transition gatherer."""

from typing import Callable

import numpy as np

from cardio_rl.agent import Agent
from cardio_rl.tracers.tracer import Tracer
from cardio_rl.types import Environment, Transition


class RayGatherer:
    """Default gatherer that steps through a single environment."""

    def __init__(
        self,
        env_fn: Callable,
        tracer: Tracer = Tracer(),
    ) -> None:
        """TODO."""
        self.env: Environment = env_fn()
        self.tracer = tracer
        self.state, _ = self.env.reset(seed=np.random.randint(np.iinfo(np.int32).max))
        self.t = 0
        self.ep_steps = 0
        self.n_envs = 1

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> tuple[list[Transition], list[float], list[int], int]:
        """TODO."""
        episodes_done = 0
        t_done = []
        ep_rew = []

        for _ in range(length):
            self.t += 1
            self.ep_steps += 1
            a, ext = agent.step(self.state)
            next_state, r, term, trun, info = self.env.step(a)
            done = term or trun

            transition = {"s": self.state, "a": a, "r": r, "s_p": next_state, "d": done}
            ext = agent.view(transition, ext)
            transition.update(ext)

            self.tracer.append(transition)

            self.state = next_state
            if done:
                ep_rew.append(info["episode"]["r"][0])
                t_done.append(self.t)
                episodes_done += 1
                self.state, _ = self.env.reset(
                    seed=np.random.randint(np.iinfo(np.int32).max)
                )
                agent.terminal()

        data = self.tracer.pop() if self.tracer.ready else []
        return data, ep_rew, t_done, episodes_done

    def reset(self) -> None:
        """Reset by clearing both buffers and reset the environment."""
        self.tracer.reset()
        self.t = 0
        self.ep_steps = 0
        self.state, _ = self.env.reset(seed=np.random.randint(np.iinfo(np.int32).max))

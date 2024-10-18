import itertools
from collections import deque
from typing import Deque

import numpy as np

from cardio_rl.agent import Agent
from cardio_rl.types import Environment, Transition


class Gatherer:
    """The gatherer is the primary component in Cardio and serves the purpose
    of stepping through the environment directly with a provided agent, or a
    random policy.

    The gatherer has two buffers that are used to package the
    transitions for the Runner in the desired manner. The step buffer
    collects transitions optained from singular environment steps and
    has a capacity equal to n. When the step buffer is full, it
    transforms its elements into one n-step transition and adds that
    transition to the transition buffer.

    Attributes:
        n_step (int, optional): Number of environment steps to store
            per-transition. Defaults to 1.
        transition_buffer (deque): Double ended queue used to store processed transitions, such as n-step transitions.
        step_buffer (deque): Double ended queue used to store individual environment transitions.
        state (np.ndarray): The current state of the environment.
    """

    def __init__(
        self,
        n_step: int = 1,
    ) -> None:
        """Initialises the gatherer, creating empty transition and step
        buffers.

        Args:
            n_step (int, optional): Number of environment steps to store
                per-transition. Defaults to 1.
        """
        self.n_step = n_step
        self.transition_buffer: Deque = deque()
        self.step_buffer: Deque = deque(maxlen=n_step)

    def init_env(self, env: Environment):
        """Pass the environment into gatherer and reset it to initialise the
        first state of the episode.

        Args:
            env (Environment): The gymnasium environment used within
                the gatherer.
        """
        self.env = env
        self.state, _ = self.env.reset()

    def step(
        self,
        agent: Agent,
        length: int,
    ) -> list[Transition]:
        """Step through the environment with the agent for a given number of
        environment steps, adding each to the step buffer. Once the step buffer
        is full, convert it to a transition and store them in the transition
        buffer. Finally, return the stored transitions, if any.

        Args:
            agent (Agent): The agent to step through environment with.
            length (int): The length of the rollout. If set to -1
                it performs one episode.

        Returns:
            list[Transition]: The contents of the transition buffer
                as a list.
        """
        iterable = iter(range(length)) if length > 0 else itertools.count()
        for _ in iterable:
            a, ext = agent.step(self.state)
            next_state, r, term, trun, _ = self.env.step(a)
            done = term or trun

            transition = {"s": self.state, "a": a, "r": r, "s_p": next_state, "d": done}
            ext = agent.view(transition, ext)
            transition.update(ext)

            self.step_buffer.append(transition)

            if len(self.step_buffer) == self.n_step:
                step = {
                    "s": self.step_buffer[0]["s"],
                    "a": self.step_buffer[0]["a"],
                    "r": np.array([step["r"] for step in self.step_buffer]),
                    "s_p": self.step_buffer[-1]["s_p"],
                    "d": self.step_buffer[-1]["d"],
                }
                for key, value in self.step_buffer[0].items():
                    if key not in ["s", "a", "r", "s_p", "d"]:
                        step.update({key: value})

                self.transition_buffer.append(step)

            self.state = next_state
            if done:
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
        """Completely reset the gatherer by clearing the transition and step
        buffer, and resetting the environment and current state."""
        self.step_buffer.clear()
        self.transition_buffer.clear()
        self.state, _ = self.env.reset()

    def _flush_step_buffer(self) -> None:
        """When using n-step transitions and reaching a terminal state, use the
        remaining individual steps in the step_buffer to not waste information
        i.e. iterate through states and pad reward.

        Ignore first step as that has already been added to transition
        buffer.
        """

        remainder = len(self.step_buffer)
        diff = self.n_step - remainder
        if remainder < self.n_step:
            start = 0
        else:
            start = 1

        for i in range(start, remainder):
            temp = list(self.step_buffer)[i:]
            pad = [0.0] * (i + diff)  # Ensures reward seq length is fixed to n_steps
            step = {
                "s": temp[0]["s"],
                "a": temp[0]["a"],
                "r": np.array([step["r"] for step in temp] + pad),
                "s_p": temp[-1]["s_p"],
                "d": temp[-1]["d"],
            }

            self.transition_buffer.append(step)

import copy
import logging
from typing import Optional

from gymnasium import Env
from tqdm import trange

import cardio_rl as crl
from cardio_rl import Agent, Gatherer
from cardio_rl.types import Transition


class BaseRunner:
    """
    The Vehicles object contains lots of vehicles

    Args:
        env (Env): The arg is used for...
        agent (Agent):
        rollout_len (int):
        warmup_len (int):
        n_step (int):
        gatherer (Optional[Gatherer]):

    Attributes:
        eval_env (Env):
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        rollout_len: int = 1,
        warmup_len: int = 0,
        n_step: int = 1,
        gatherer: Optional[Gatherer] = None,
    ) -> None:
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.rollout_len = rollout_len
        self.warmup_len = warmup_len

        if gatherer is None:
            self.gatherer: Gatherer = Gatherer(n_step=n_step)
        else:
            self.gatherer = gatherer

        self.n_step = n_step
        self.agent = agent

        self.gatherer._init_env(self.env)
        self.burn_in_len = 0
        if self.burn_in_len:
            self._burn_in()  # TODO: implement argument
        self.gatherer.reset()

        # Logging should only start now
        if self.warmup_len:
            self._warm_start()

    def _burn_in(self) -> None:
        """Step through environment with random agent, e.g. to
        initialise observation normalisation. Gatherer is
        reset afterwards.
        """
        self._rollout(self.burn_in_len, Agent(self.env))

    def _rollout(self, steps: int, agent: Agent) -> tuple[list[Transition], int]:
        """Internal method to step through environment for a
        provided number of steps. Return the collected
        transitions and how many were collected.

        Args:
            steps (int): Number of steps to take in environment
            agent (Agent): Can optionally pass a specific agent to
                step through environment with

        Returns:
            rollout_transitions (Transition): stacked Transitions
                from environment
            num_transitions (int): number of Transitions collected
        """
        rollout_transitions = self.gatherer.step(agent, steps)
        num_transition = len(rollout_transitions)
        return rollout_transitions, num_transition

    def _warm_start(self):
        """Step through environment with freshly initialised
        agent, to collect transitions before training via
        the _rollout internal method.
        """

        self._rollout(self.warmup_len, self.agent)
        logging.info("### Warm up finished ###")

    def step(self, agent: Optional[Agent] = None) -> list[Transition]:
        """Main method to step through environment with
        agent, to collect transitions and pass them to your
        agent's update function.

        Args:
            agent (Agent): Can optionally pass a specific agent to
                step through environment with

        Returns:
            batch (list[Transition]): A list of Transitions
                sampled from the environment

        """

        agent = agent if self.agent is None else self.agent
        rollout_batch, num_transitions = self._rollout(self.rollout_len, agent)  # type: ignore
        del num_transitions
        batch = [self.transform_batch(rollout_batch)]
        return batch

    def run(
        self,
        rollouts: int = 1_000_000,
        eval_interval: int = 0,
        eval_episodes: int = 0,
    ) -> None:
        """Iteratively run runner.step() for self.rollout_len
        and pass the batched data through to the agents update
        step.

        Args:
            rollouts (int): The number of rollouts of length self.rollout_len to
                undertake
            eval_interval (int): How many rollouts to take in between evaluations
            eval_episodes (int): How many episodes to perform during evaluation
        """

        for i in trange(rollouts):
            data = self.step()
            self.agent.update(data)  # type: ignore

    def evaluate(self, episodes: int) -> dict:
        """To be returned to when updating logging

        Args:
            episodes (int): number of episodes to evaluate over
        """

        return {}
        # metrics = {"return": np.zeros(episodes), "length": np.zeros(episodes)}
        # for e in range(episodes):
        #     state, _ = self.eval_env.reset()
        #     returns = 0.0
        #     steps: int = 0
        #     while True:
        #         action, _ = self.agent.eval_step(state)
        #         next_state, reward, done, trun, _ = self.eval_env.step(action)
        #         returns += reward  # type: ignore
        #         steps += 1
        #         state = next_state
        #         if done or trun:
        #             metrics["return"][e] = returns
        #             metrics["length"][e] = steps
        #             break

        # return metrics

    def transform_batch(self, batch: list[Transition]) -> Transition:
        """Perform some transformation of a given list of Transitions

        Args:
            batch (list[Transition]): A list of the Transitions to be
                stacked via crl.tree.stack

        Returns:
            transformed_batch (Transition): Transition that is the
                stacked input list of Transitions
        """

        transformed_batch = crl.tree.stack(batch)
        return transformed_batch

    def update_agent(self, new_agent: Agent):
        """Update the Agent being used in the Runner"""

        self.agent = new_agent

    def reset(self) -> None:
        """Perform any necessary resets, such as for the gatherer"""

        self.gatherer.reset()

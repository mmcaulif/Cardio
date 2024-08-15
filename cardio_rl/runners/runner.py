import copy
import logging
import time
from typing import Callable, Optional

import jax
from gymnasium import Env
from gymnasium.experimental.vector import VectorEnv
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

import cardio_rl as crl
from cardio_rl import Agent, Gatherer
from cardio_rl.types import Transition

logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt=" %I:%M:%S %p",
    level=logging.INFO,
)


class BaseRunner:
    """_summary_"""

    def __init__(
        self,
        env: Env | VectorEnv,
        agent: Optional[Agent] = None,
        rollout_len: int = 1,
        warmup_len: int = 0,
        n_step: int = 1,
        eval_env: Optional[Env] = None,
        gatherer: Optional[Gatherer] = None,
    ) -> None:
        """_summary_

        Args:
            env (Env | VectorEnv): _description_
            agent (Optional[Agent], optional): _description_. Defaults to None.
            rollout_len (int, optional): _description_. Defaults to 1.
            warmup_len (int, optional): _description_. Defaults to 0.
            n_step (int, optional): _description_. Defaults to 1.
            eval_env (Env, optional): _description_. Defaults to None.
            gatherer (Optional[Gatherer], optional): _description_. Defaults to None.
        """
        self.env = env
        # if isinstance(env, VectorEnv):
        #     self.n_envs: int = env.num_envs
        # else:
        #     self.n_envs: int = 1

        self.n_envs = 1 if not isinstance(env, VectorEnv) else env.num_envs

        self.agent = agent
        self.rollout_len = rollout_len
        self.warmup_len = warmup_len

        if gatherer is None:
            self.gatherer: Gatherer = Gatherer(n_step=n_step)
        else:
            self.gatherer = gatherer

        self.n_step = n_step

        if eval_env is None:
            self.eval_env = copy.deepcopy(env)
        else:
            self.eval_env = eval_env

        self.logger = logging.getLogger()
        self.initial_time = time.time()

        # Initialise components
        self.gatherer.init_env(self.env)
        self.burn_in_len = 0
        if self.burn_in_len:
            self._burn_in()  # TODO: implement argument
        self.gatherer.reset()

        if self.warmup_len:
            self._warm_start()

    def _burn_in(self) -> None:
        """Step through environment with random agent, e.g. to
        initialise observation normalisation. Gatherer is
        reset afterwards.
        """
        self._rollout(self.burn_in_len, Agent(self.env))

    def _warm_start(self):
        """Step through environment with freshly initialised
        agent, to collect transitions before training via
        the _rollout internal method.
        """
        agent = self.agent or crl.Agent(self.env)  # Needs to be ordered like this!
        rollout_transitions, num_transitions = self._rollout(self.warmup_len, agent)
        logging.info("### Warm up finished ###")
        return rollout_transitions, num_transitions

    def _rollout(
        self,
        steps: int,
        agent: Optional[Agent] = None,
        transform: Optional[Callable] = None,
    ) -> tuple[Transition, int]:
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
        rollout_transitions = self.gatherer.step(agent, steps)  # type: ignore
        num_transitions = len(rollout_transitions)
        if num_transitions:
            rollout_transitions = self.transform_batch(rollout_transitions, transform)  # type: ignore
        return rollout_transitions, num_transitions  # type: ignore

    def step(
        self, transform: Optional[Callable] = None, agent: Optional[Agent] = None
    ) -> list[Transition]:
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
        rollout_batch, num_transitions = self._rollout(
            self.rollout_len, agent, transform
        )  # type: ignore
        del num_transitions
        return [rollout_batch]

    def eval(self, episodes: int, agent: Optional[Agent] = None) -> float:
        agent = agent if self.agent is None else self.agent
        avg_returns = 0.0
        for _ in range(episodes):
            s, _ = self.eval_env.reset()
            returns = 0.0
            while True:
                # TODO: fix this mypy issue
                a = agent.eval_step(s)  # type: ignore
                s_p, r, d, t, _ = self.eval_env.step(a)
                returns += r
                s = s_p
                if d or t:
                    avg_returns += returns
                    break

        return avg_returns / episodes

    def run(
        self, rollouts: int = 1_000_000, eval_freq: int = 1_000, eval_episodes: int = 10
    ) -> None:
        """Iteratively run runner.step() for self.rollout_len
        and pass the batched data through to the agents update
        step.

        Args:
            rollouts (int): The number of rollouts of length self.rollout_len to
                undertake
            eval_freq (int): How many rollouts to take in between evaluations
            eval_episodes (int): How many episodes to perform during evaluation
        """

        for t in trange(rollouts):
            data = self.step()
            updated_data = self.agent.update(data)  # type: ignore
            if updated_data:
                self.update(updated_data)
            if t % eval_freq == 0 and t > 0:
                avg_returns = self.eval(eval_episodes, self.agent)
                with logging_redirect_tqdm():
                    env_steps = (self.n_envs * t * self.rollout_len) + self.warmup_len
                    curr_time = round(time.time() - self.initial_time, 2)
                    metrics = {
                        "Timesteps": env_steps,
                        # "Episodes": self.episodes,    # TODO: find a way to implement this
                        "Avg eval returns": avg_returns,
                        "Time passed": curr_time,
                        "Env steps per second": int(env_steps / curr_time),
                    }
                    self.logger.info(metrics)

    def transform_batch(
        self, batch: list[Transition], transform: Optional[Callable] = None
    ) -> Transition:
        """Perform some transformation of a given list of Transitions

        Args:
            batch (list[Transition]): A list of the Transitions to be
                stacked via crl.tree.stack

        Returns:
            transformed_batch (Transition): Transition that is the
                stacked input list of Transitions
        """

        transformed_batch = crl.tree.stack(batch)
        if transform:
            transformed_batch = jax.tree.map(transform, transformed_batch)
        return transformed_batch

    def update_agent(self, new_agent: Agent):
        """Update the Agent being used in the Runner"""
        self.agent = new_agent

    def reset(self) -> None:
        """Perform any necessary resets, such as for the gatherer"""
        self.gatherer.reset()

    def update(self, data: dict) -> None:
        """Perform any necessary updates, such as for the replay buffer"""
        del data
        pass

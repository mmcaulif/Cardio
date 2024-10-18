import copy
import time
from typing import Callable, Optional

import jax
import numpy as np
from gymnasium import Env
from gymnasium.experimental.vector import VectorEnv
from gymnasium.wrappers import record_episode_statistics
from tqdm import trange

import cardio_rl as crl
from cardio_rl import Agent, Gatherer
from cardio_rl.loggers import BaseLogger
from cardio_rl.types import Environment, Transition


class BaseRunner:
    """The runner is the high level orchestrator that deals with the different
    components and data, it contains a gatherer, your agent and any replay
    buffer you might have. The runner calls the gatherer's step function as
    part its own step function, or as part of its built in warmup (for
    collecting a large amount of initial data with your agent) and burnin (for
    randomly stepping through an environment, not collecting data, such as for
    initialising normalisation values) methods. The runner can either be used
    via its run method (which iteratively calls the runner.step and the
    agent.update methods) or with each mothod individually with its step method
    if you'd like more finegrained control.

    Attributes:
        env (Environment): The gym environment used to collect transitions and
            train the agent.
        n_envs (int): The number of environments, only > 1 if env is a VectorEnv.
        agent (Optional[Agent], optional): The agent stored within the runner, used
            by the step and run methods. Defaults to None.
        rollout_len (int, optional): Number of environment steps to perform as part
            of the step method. Defaults to 1.
        warmup_len (int, optional): Number of steps to perform with the agent before
            regular rollouts begin. Defaults to 0.
        n_step (int, optional): Number of environment steps to store within a single
            transition. Defaults to 1.
        eval_env (Optional[Env], optional): An optional separate environment to
            be used for evaluation, must not be a VectorEnv. Defaults to None.
        gatherer (Optional[Gatherer], optional): An optional gatherer to be used by
            the runner. Defaults to None.
        _initial_time (float): The time in seconds when the runner was initialised.
    """

    def __init__(
        self,
        env: Environment,
        agent: Optional[Agent] = None,
        rollout_len: int = 1,
        warmup_len: int = 0,
        n_step: int = 1,
        eval_env: Optional[Env] = None,
        logger: Optional[BaseLogger] = None,
        gatherer: Optional[Gatherer] = None,
    ) -> None:
        """Initialises a generic runner, parent class of OnPolicyRunner and
        OffPolicyRunner, which should be used instead of this. This base class is
        extensible and contains all the necessary attributes for both collecting
        transitions with the gatherer and formatting them correctly for later use by
        the agent. Data is provided to the runner and from the runner to the agent as
        a dictionary with the following keys: s, a, r, s_p and d, representing the state,
        action, reward, next state and done features of an MDP transition.

        Args:
            env (Environment): The gym environment used to collect transitions and
                train the agent.
            agent (Optional[Agent], optional): The agent stored within the runner, used
                by the step and run methods. Defaults to None.
            rollout_len (int, optional): Number of environment steps to perform as part
                of the step method. Defaults to 1.
            warmup_len (int, optional): Number of steps to perform with the agent before
                regular rollouts begin. Defaults to 0.
            n_step (int, optional): Number of environment steps to store within a single
                transition. Defaults to 1.
            eval_env (Optional[gym.Env], optional): An optional separate environment to
                be used for evaluation, must not be a VectorEnv. Defaults to None.
            gatherer (Optional[Gatherer], optional): An optional gatherer to be used by
                the runner. Defaults to None.

        Raises:
            TypeError: Trying to use a VectorEnv with n_step > 1.
        """
        if isinstance(env, VectorEnv) and n_step > 1:
            raise TypeError("VectorEnv's not yet compatible with n_step > 1")

        self.env = env
        self.n_envs = 1 if not isinstance(env, VectorEnv) else env.num_envs

        self.agent = agent
        self.rollout_len = rollout_len
        self.warmup_len = warmup_len
        self.gatherer = gatherer or Gatherer(n_step=n_step)
        self.n_step = n_step
        self.eval_env = eval_env or copy.deepcopy(env)
        self.eval_env = record_episode_statistics.RecordEpisodeStatistics(self.eval_env)  # type: ignore

        self._initial_time = time.time()

        self.logger = logger or crl.loggers.BaseLogger()

        # Initialise components
        self.gatherer.init_env(self.env)
        self.burn_in_len = 0
        if self.burn_in_len:
            self._burn_in()  # TODO: implement argument
        self.gatherer.reset()

        if self.warmup_len:
            self._warm_start()

    def _burn_in(self) -> None:
        """Step through environment with random agent, e.g. to initialise
        observation normalisation.

        Gatherer is reset afterwards.
        """
        self._rollout(self.burn_in_len, Agent(self.env))

    def _warm_start(self) -> tuple[Transition, int]:
        """Step through environment with freshly initialised agent, to collect
        transitions before training via the agents update method.

        Returns:
            Transition: stacked Transitions from environment
            int: number of Transitions collected
        """
        agent = self.agent or crl.Agent(self.env)  # Needs to be ordered like this!
        rollout_transitions, num_transitions = self._rollout(self.warmup_len, agent)
        self.logger.terminal("### Warm up finished ###")
        return rollout_transitions, num_transitions

    def _rollout(
        self,
        steps: int,
        agent: Agent,
        transform: Optional[Callable] = None,
    ) -> tuple[Transition, int]:
        """Internal method to step through environment for a provided number of
        steps. Return the collected transitions and how many were collected.

        Args:
            steps (int): Number of steps to take in environment.
            agent (Optional[Agent], optional): Can optionally pass a
                specific agent to step through environment with. Defaults
                to None and uses internal agent.
            transform (Optional[Callable]. optional): An optional function
                to use on the stacked Transitions received during the
                rollout. Defaults to None.

        Returns:
            Transition: stacked Transitions from environment.
            int: number of Transitions collected
        """
        rollout_transitions = self.gatherer.step(agent, steps)
        num_transitions = len(rollout_transitions)
        if num_transitions:
            rollout_transitions = self.transform_batch(rollout_transitions, transform)  # type: ignore
        return rollout_transitions, num_transitions  # type: ignore

    def step(
        self, transform: Optional[Callable] = None, agent: Optional[Agent] = None
    ) -> Transition | list[Transition]:
        """Main method to step through environment with agent, to collect
        transitions and pass them to your agent's update function.

        Args:
            transform (Optional[Callable]. optional): An optional function
                to use on the stacked Transitions received during the
                rollout. Defaults to None.
            agent (Optional[Agent], optional): Can optionally pass a
                specific agent to step through environment with. Defaults
                to None and uses internal agent.

        Returns:
            Transition | list[Transition]: stacked Transitions from environment.
        """

        agent = agent or self.agent
        assert agent is not None
        rollout_batch, num_transitions = self._rollout(
            self.rollout_len, agent, transform
        )
        del num_transitions
        return rollout_batch

    def eval(
        self, rollouts: int, episodes: int, agent: Optional[Agent] = None
    ) -> float:
        """Step through the eval_env for a given number of episodes using the
        agents eval_step method, recording the episodic return and calculating
        the average over all episodes.

        Args:
            episodes (int): The number of episodes to perform with the agent.
            agent (Optional[Agent], optional): Can optionally pass a
                specific agent to step through environment with. Defaults
                to None and uses internal agent.

        Returns:
            float: Average of the total episodic return received over the
                evaluation episodes.
        """
        agent = agent or self.agent
        avg_r = 0.0
        avg_l = 0.0
        sum_t = 0.0
        for _ in range(episodes):
            s, _ = self.eval_env.reset()
            while True:
                # TODO: fix below mypy issue
                a = agent.eval_step(s)  # type: ignore
                s_p, _, term, trun, info = self.eval_env.step(a)
                done = term or trun
                s = s_p
                if done:
                    avg_r += info["episode"]["r"]
                    avg_l += info["episode"]["l"]
                    sum_t += info["episode"]["t"]
                    break

        avg_r = float(avg_r / episodes)
        avg_l = float(avg_l / episodes)
        sum_t = float(sum_t)
        # with logging_redirect_tqdm():

        env_steps = (self.n_envs * rollouts * self.rollout_len) + self.warmup_len
        curr_time = round(time.time() - self._initial_time, 2)
        metrics = {
            "Timesteps": env_steps,
            "Training steps": rollouts,
            # "Episodes": self.episodes,    # TODO: find a way to implement this
            "Avg eval returns": round(avg_r, 2),
            "Avg eval episode length": avg_l,
            "Time passed": curr_time,
            "Evaluation time": round(sum_t, 4),
            "Steps per second": int(env_steps / curr_time),
        }
        # logging.info(metrics)
        self.logger.log(metrics)

        return avg_r

    def run(
        self, rollouts: int, eval_freq: int = 1_000, eval_episodes: int = 10
    ) -> float:
        """Iteratively run runner.step() for self.rollout_len and pass the
        batched data through to the agents update step. Stops and calls
        self.eval every eval_freq with eval_episodes. After all rollouts are
        taken, a final evaluation step is called and the average episodic
        returns from the final evaluation step are returned by this method.

        Args:
            rollouts (int): The number of rollouts of length self.rollout_len to
                perform.
            eval_freq (int): How many rollouts to take in between evaluations.
            eval_episodes (int): How many episodes to perform during evaluation.

        Returns:
            float: Average episodic returns from the final evaluation step.
        """
        self.logger.terminal("Performing initial evaluation")
        _ = self.eval(
            0, eval_episodes, self.agent
        )  # TODO: have this before the warmup?

        for t in trange(rollouts):
            data = self.step()
            updated_data = self.agent.update(data)  # type: ignore
            if updated_data:
                self.update(updated_data)
            if t % eval_freq == 0 and t > 0:
                self.eval(t, eval_episodes, self.agent)

        self.logger.terminal("Performing final evaluation")
        avg_returns = self.eval(t, eval_episodes, self.agent)
        return avg_returns

    def transform_batch(
        self, batch: list[Transition], transform: Optional[Callable] = None
    ) -> Transition:
        """Stack a list of Transitions via cardio_rl.tree.stack followed by an
        optional transformation.

        Args:
            batch (list[Transition]): A list of the Transitions to be stacked.

        Returns:
            Transition: The stacked input list of Transitions.
        """

        transformed_batch = crl.tree.stack(batch)

        def _expand(arr):
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, -1)

            return arr

        transformed_batch = jax.tree.map(_expand, transformed_batch)

        if transform:
            transformed_batch = jax.tree.map(transform, transformed_batch)
        return transformed_batch

    def update_agent(self, new_agent: Agent):
        """Update the internl agent being used in the Runner.

        Args:
            new_agent (Agent): An agent to replace the current internal
                agent.
        """
        self.agent = new_agent

    def reset(self) -> None:
        """Perform any necessary resets, such as for the gatherer."""
        self.gatherer.reset()

    def update(self, data: dict) -> None:
        """Perform any necessary updates, does nothing in the BaseRunner.

        data (dict): A dictionary containing the indices and keys/values
        to be updated.
        """
        del data
        pass

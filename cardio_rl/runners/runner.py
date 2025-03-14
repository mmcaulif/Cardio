"""High level orchestrator of an Environment and Cardio components."""

from __future__ import annotations

import copy
import time
import warnings
from typing import Callable

import jax
import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from tqdm import trange

import cardio_rl as crl
from cardio_rl import Agent, Gatherer
from cardio_rl.buffers.base_buffer import BaseBuffer
from cardio_rl.evaluation import evaluate_agent
from cardio_rl.loggers import BaseLogger
from cardio_rl.types import Environment, Transition


class Runner:
    """Cardio's primary component, the Runner."""

    def __init__(
        self,
        env: Environment,
        agent: Agent | None = None,
        rollout_len: int = 1,
        warmup_len: int = 0,
        n_step: int = 1,
        eval_env: Env | None = None,
        burnin_len: int = 0,
        buffer: BaseBuffer | None = None,
        logger: BaseLogger | None = None,
        gatherer: Gatherer | None = None,
    ) -> None:
        """Initialise the primary Runner.

        Initialise the Runner. Contains the agent, gatherer, optional
        buffer and a logger, which are the necessary components for
        both collecting transitions with the gatherer and formatting
        them correctly for later use by the agent. Data is provided to
        the runner and from the runner to the agent as a dictionary
        with the following keys: s, a, r, s_p and d, as well as any
        extra specifications from the agent. The runner calls the
        gatherer's step function as part its own step function, or as
        part of its built in warmup (for collecting a large amount of
        initial data with your agent) and burnin (for randomly stepping
        through an environment, not collecting data, such as for
        initialising normalisation values) methods. The runner can
        either be used via its run method (which iteratively calls the
        runner.step and the agent.update methods) or with each mothod
        individually with its step method if you'd like more
        finegrained control.

        Args:
            env (Environment): The gym environment used to collect
                transitions and train the agent.
            agent (Agent | None, optional): The agent stored within the
                runner, used by the step and run methods. If set to
                None, a random Agent is used. Defaults to None.
            rollout_len (int, optional): Number of environment steps to
                perform as part of the step method. Defaults to 1.
            warmup_len (int, optional): Number of steps to perform with
                the agent before regular rollouts begin. Defaults to 0.
            n_step (int, optional): Number of environment steps to
                store within a single transition. Defaults to 1.
            eval_env (gym.Env | None, optional): An optional separate
                environment to be used for evaluation, must not be a
                VectorEnv. Defaults to None.
            burnin_len (int, optional): Number of untracked environment
                steps to take before warmup, such as to initialise
                normalisation. Defaults to 0.
            buffer (BaseBuffer | None, optional): Can pass a buffer
                which stores data and is randomly sampled when calling
                runner.step. Defaults to None.
            logger (BaseLogger | None, optional): The logger used
                during evaluations, is set to None uses the BaseLogger
                which prints to terminal and writes to a file in a
                logs directory. Defaults to None.
            gatherer (Gatherer | None, optional): An optional gatherer
                to be used by the runner. If set to None the default
                Gatherer will be used. Defaults to None.

        Raises:
            TypeError: Trying to use a VectorEnv with n_step > 1.
        """
        if isinstance(env, VectorEnv) and n_step > 1:
            raise TypeError("VectorEnv's not yet compatible with n_step > 1")

        self.env = env

        if not isinstance(self.env, RecordEpisodeStatistics):
            self.env = RecordEpisodeStatistics(env)

        self.agent = agent
        self.rollout_len = rollout_len
        self.warmup_len = warmup_len
        self.gatherer = gatherer or Gatherer(n_step=n_step)
        self.n_step = n_step
        self.eval_env = eval_env

        if self.eval_env is not None and not isinstance(
            self.eval_env, RecordEpisodeStatistics
        ):
            self.eval_env = RecordEpisodeStatistics(self.eval_env)

        self.buffer = buffer

        self._initial_time = time.time()
        self.train_rew: list[float] = []
        self.rollout_train_rew: list[float] = []
        self.t_completed: list[int] = []
        self.total_episodes = 0
        self.rollout_ep_completed = 0
        self.agent_metrics: list[dict] = []

        self.logger = logger or crl.loggers.BaseLogger()

        # Initialise components
        self.gatherer.init_env(self.env)
        self.burnin_len = burnin_len
        if self.burnin_len > 0:
            self._burn_in()

        if self.warmup_len > 0:
            self._warm_start()

    def _burn_in(self) -> None:
        """Step through environment with random agent.

        Performs a fixed number of environment steps such as to
        initialise observation normalisation. Gatherer is reset
        afterwards.
        """
        self.gatherer.step(crl.Agent(self.env), self.burnin_len)
        self.gatherer.reset()
        self.logger.terminal("### Burn in finished ###")

    def _warm_start(self) -> tuple[Transition, int]:
        """Step through environment with an agent.

        Performs a fixed number of environment steps such as to collect
        transitions before training via the agents update method.

        Returns:
            Transition: stacked Transitions from environment.
            int: number of Transitions collected,
        """
        agent = self.agent or crl.Agent(self.env)  # Needs to be ordered like this!
        rollout_transitions, num_transitions = self._rollout(self.warmup_len, agent)
        self.logger.terminal("### Warm up finished ###")
        if self.buffer is not None:
            self.buffer.store(rollout_transitions, num_transitions)
        return rollout_transitions, num_transitions

    def _rollout(
        self,
        steps: int,
        agent: Agent,
        transform: Callable | None = None,
    ) -> tuple[Transition, int]:
        """Perform rollout via the gatherer.step method.

        Internal method to step through environment for a provided
        number of steps. Return the collected transitions and how many
        were collected. Wraps the gatherer.step method and performs
        some light processing on the returned Transitions via the
        self.transform_batch method.

        Args:
            steps (int): Number of steps to take in environment.
            agent (Agent | None, optional):  Can optionally pass a
                specific agent to step through environment with, if set
                to None the internal agent will be used. Defaults to
                None.
            transform (Callable | None. optional): An optional function
                to use on the stacked Transitions received during the
                rollout. Defaults to None.

        Returns:
            Transition: stacked Transitions from environment.
            int: number of Transitions collected
        """
        rollout_transitions, ep_rew, t_completed, ep_completed = self.gatherer.step(
            agent, steps
        )

        if ep_completed > 0:
            self.train_rew += ep_rew
            self.rollout_train_rew += ep_rew
            self.t_completed += t_completed
            self.rollout_ep_completed += ep_completed
            self.total_episodes += ep_completed

        num_transitions = len(rollout_transitions)
        if num_transitions:
            rollout_transitions = self.transform_batch(rollout_transitions, transform)  # type: ignore
        return rollout_transitions, num_transitions  # type: ignore

    def step(
        self, transform: Callable | None = None, agent: Agent | None = None
    ) -> Transition | list[Transition]:
        """Perform a rollout and return or store the Transitions.

        Main method to step through environment with agent, to collect
        transitions and pass them to your agent's update function.
        Calls the self._rollout method and either directly returns the
        Transitions or places them in a buffer which is then sampled
        from.

        Args:
            transform (Callable | None. optional): An optional function
                to use on the stacked Transitions received during the
                rollout. Defaults to None.
            agent (Agent | None, optional): Can optionally pass a
                specific agent to step through environment with, if set
                to None the internal agent will be used. Defaults to
                None.

        Returns:
            Transition | list[Transition]: stacked Transitions from
                environment.
        """
        agent = agent or self.agent
        assert agent is not None
        rollout_transitions, num_transitions = self._rollout(
            self.rollout_len, agent, transform
        )

        if self.buffer is not None:
            if num_transitions:
                self.buffer.store(rollout_transitions, num_transitions)
            return self.buffer.sample()

        return rollout_transitions

    def eval(self, episodes: int, agent: Agent | None = None) -> tuple[float, float]:
        """Evaluate an agent's performance.

        Step through the eval_env for a given number of episodes using
        agent.eval_step, recording the episodic return and calculating
        the average over all episodes.

        Args:
            episodes (int): The number of episodes to perform with the
                agent.
            agent (Agent | None, optional): Can optionally pass a
                specific agent to evaluate, if set to None the internal
                agent will be used. Defaults to None.

        Returns:
            float: Average of the total episodic return received over
                the evaluation episodes.
        """
        if self.eval_env is None:
            eval_env = copy.deepcopy(self.env)
        else:
            eval_env = self.eval_env  # type: ignore

        agent = agent or self.agent
        assert agent is not None

        eval_returns, eval_t = evaluate_agent(
            eval_env, agent, episodes, return_episode_rewards=True
        )

        mean_returns = np.array(eval_returns).mean().item()
        std_returns = np.array(eval_returns).std().item()

        train_steps = (self.gatherer.t - self.warmup_len) // (
            self.rollout_len * self.gatherer.n_envs
        )
        curr_time = round(time.time() - self._initial_time, 2)
        metrics = {
            "Timesteps": self.gatherer.t,
            "Update steps": train_steps,
            "Avg eval return": round(mean_returns, 2),
            "Std eval return": round(std_returns, 2),
            "Time passed": curr_time,
            "Evaluation time": round(eval_t, 4),
            "Steps per second": round(self.gatherer.t / curr_time, 4),
        }
        if self.rollout_ep_completed > 0:
            metrics.update(
                {
                    "Training episodes": self.total_episodes,
                    "Avg training returns": round(
                        np.mean(self.rollout_train_rew).item(), 2
                    ),
                }
            )
            self.rollout_train_rew.clear()
            self.rollout_ep_completed = 0

        if self.agent_metrics:
            stacked_agent_metrics = crl.tree.stack(self.agent_metrics)
            avg_agent_metrics = jax.tree.map(np.mean, stacked_agent_metrics)
            avg_agent_metrics = jax.tree.map(
                lambda val: round(val, 4), avg_agent_metrics
            )
            metrics.update(avg_agent_metrics)
            self.agent_metrics.clear()

        self.logger.log(metrics)
        return mean_returns, std_returns

    def run(
        self,
        rollouts: int,
        eval_freq: int = 1_000,
        eval_episodes: int = 10,
        tqdm: bool = True,
    ) -> tuple[float, float]:
        """Primary method to train an agent.

        Iteratively run runner.step() for self.rollout_len and pass the
        batched data through to the agents update step. Stops and calls
        self.eval every eval_freq with eval_episodes. After all
        rollouts are taken, a final evaluation step is called and the
        average episodic returns from the final evaluation step are
        returned by this method.

        Args:
            rollouts (int): The number of rollouts of length
                self.rollout_len to perform.
            eval_freq (int): How many rollouts to take in between
                evaluations.
            eval_episodes (int): How many episodes to perform during
                evaluation.
            tqdm (bool): Whether to use a tqdm-style loading bar.

        Returns:
            float: Average episodic returns from the final evaluation
                step.
        """
        self.logger.terminal("Performing initial evaluation")
        _ = self.eval(
            eval_episodes, self.agent
        )  # Have this before burn-in (idk why I thought it should be before burn in...)

        _disable = not tqdm
        for n in trange(rollouts, disable=_disable):
            if n % eval_freq == 0 and n > 0:
                self.eval(eval_episodes, self.agent)
            data = self.step()
            feedback = self.agent.update(data)  # type: ignore

            if feedback:
                if isinstance(feedback, tuple):
                    agent_metrics, updated_data = feedback
                    if agent_metrics:
                        self.agent_metrics.append(agent_metrics)
                    if updated_data:
                        self.update(updated_data)
                else:
                    self.agent_metrics.append(feedback)

        self.logger.terminal("Performing final evaluation")
        avg_returns, std_returns = self.eval(eval_episodes, self.agent)
        self.logger.dump(self.train_rew, self.t_completed, self.env.spec.id)  # type: ignore
        return avg_returns, std_returns

    def transform_batch(
        self, batch: list[Transition], transform: Callable | None = None
    ) -> Transition:
        """Stack and reshape a list of Transitions.

        Stack a list of Transitions via cardio_rl.tree.stack followed
        by an optional transformation.

        Args:
            batch (list[Transition]): A list of the Transitions to be
                stacked.
            transform (Callable): An optional transform that can be
                applied to the Transitions post-stacking. Defaults
                to None.

        Returns:
            Transition: The stacked input list of Transitions.
        """
        transformed_batch = crl.tree.stack(batch)

        # TODO: unsure if this should be used as it behaves inconsistently
        # between different configurations.
        def _expand(arr):
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, -1)

            return arr

        transformed_batch = jax.tree.map(_expand, transformed_batch)

        if transform:
            transformed_batch = jax.tree.map(transform, transformed_batch)
        return transformed_batch

    def update_agent(self, new_agent: Agent):
        """Replace the internl agent.

        Update the internal agent being used in the Runner.

        Args:
            new_agent (Agent): An agent to replace the current internal
                agent.
        """
        self.agent = new_agent

    def reset(self) -> None:
        """Perform any necessary resets, such as for the gatherer."""
        self.gatherer.reset()

    def update(self, data: dict) -> None:
        """Update the data stored in the buffer, if one exists.

        update specific keys and indices in the internal table with new
        or updated data, e.g. latest priorities. Only called if the
        agent's update step returns a non-empty dictionary. Must
        provide indices via a "idxs" key. If the runner has no buffer
        no buffer, this method does nothing.

        Args:
            data (dict): A dictionary containing an "idxs" key with
                indices to update and keys with the updated values.
        """
        if self.buffer is not None:
            self.buffer.update(data)

    @classmethod
    def on_policy(
        cls,
        env: Environment,
        agent: Agent | None = None,
        rollout_len: int = 1,
        eval_env: Env | None = None,
        burnin_len: int = 0,
        logger: BaseLogger | None = None,
    ) -> Runner:
        """Construct a Runner for on-policy learning.

        _extended_summary_ <- TODO

        Args:
            env (Environment): The gym environment used to collect
                transitions and train the agent.
            agent (Agent | None, optional): The agent stored within the
                runner, used by the step and run methods. If set to
                None, a random Agent is used. Defaults to None.
            rollout_len (int, optional): Number of environment steps to
                perform as part of the step method. Defaults to 1.
            eval_env (Env | None, optional): An optional separate
                environment to be used for evaluation, must not be a
                VectorEnv. Defaults to None.
            burnin_len (int, optional): Number of untracked environment
                steps to take before warmup, such as to initialise
                normalisation. Defaults to 0.
            logger (BaseLogger | None, optional): The logger used
                during evaluations, is set to None uses the BaseLogger
                which prints to terminal and writes to a file in a
                logs directory. Defaults to None.

        Returns:
            Runner: A Runner designed for on-policy algorithms.
        """
        gatherer = (
            crl.VectorGatherer() if isinstance(env, VectorEnv) else crl.Gatherer()
        )
        return cls(
            env=env,
            agent=agent,
            rollout_len=rollout_len,
            warmup_len=0,
            n_step=1,
            eval_env=eval_env,
            burnin_len=burnin_len,
            logger=logger,
            gatherer=gatherer,
        )

    @classmethod
    def off_policy(
        cls,
        env: Environment,
        agent: Agent | None = None,
        buffer_kwargs: dict = {},
        rollout_len: int = 1,
        warmup_len: int = 10_000,
        eval_env: Env | None = None,
        burnin_len: int = 0,
        extra_specs: dict | None = None,
        buffer: BaseBuffer | None = None,
        logger: BaseLogger | None = None,
    ) -> Runner:
        """Construct a Runner for off-policy learning.

        _extended_summary_ <- TODO
        Mention now the buffer uses the buffers n_step attribute.

        Args:
            env (Environment): The gym environment used to collect
                transitions and train the agent.
            agent (Agent | None, optional): The agent stored within the
                runner, used by the step and run methods. If set to
                None, a random Agent is used. Defaults to None.
            buffer_kwargs (dict, optional): A dictionary with any
                keyword arguments for the TreeBuffer that is used by
                default. Defaults to {}.
            rollout_len (int, optional): Number of environment steps to
                perform as part of the step method. Defaults to 1.
            warmup_len (int, optional): Number of steps to perform with
                the agent before regular rollouts begin. Defaults to
                10_000.
            eval_env (gym.Env | None, optional): An optional separate
                environment to be used for evaluation, must not be a
                VectorEnv. Defaults to None.
            burnin_len (int, optional): Number of untracked environment
                steps to take before warmup, such as to initialise
                normalisation. Defaults to 0.
            extra_specs (dict | None, optional):  Additional entries to
                add to the replay buffer. Values must correspond to the
                shape. Defaults to None. Defaults to None.
            buffer (BaseBuffer | None, optional): Can pass a buffer
                which stores data and is randomly sampled when calling
                runner.step, if set to None a default TreeBuffer will
                be used with kwargs equal to those defined in the
                buffer_kwargs argument. Defaults to None.
            logger (BaseLogger | None, optional): The logger used
                during evaluations. If set to None, uses the BaseLogger
                which prints to terminal and writes to a file in a
                logs directory. Defaults to None.

        Raises:
            TypeError: Trying to pass a VectorEnv environment to this
                runner.

        Returns:
            Runner: A Runner designed for off-policy algorithms.
        """
        if isinstance(env, VectorEnv):
            raise TypeError("VectorEnv's not yet compatible with off-policy runner")

        if buffer is not None:
            warnings.warn(
                "Provided a buffer, ignoring the extra_specs and buffer_kwargs arguments"
            )
            buffer = buffer
        else:
            buffer = crl.buffers.TreeBuffer(
                env, extra_specs=extra_specs, **buffer_kwargs
            )

        return cls(
            env=env,
            agent=agent,
            rollout_len=rollout_len,
            warmup_len=warmup_len,
            n_step=buffer.n_steps,
            eval_env=eval_env,
            burnin_len=burnin_len,
            buffer=buffer,
            logger=logger,
        )

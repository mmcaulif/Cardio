from typing import Callable, Optional

from gymnasium import Env
from gymnasium.experimental.vector import VectorEnv

from cardio_rl import Agent, BaseRunner, Gatherer
from cardio_rl.buffers import BaseBuffer, TreeBuffer
from cardio_rl.types import Environment, Transition


class OffPolicyRunner(BaseRunner):
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

    TODO: attribute decriptions should be different to the argument descriptions when
    applicable, i.e. for the gathere, the attribute description should describe the
    gatherer and what it does but the argument dicstring should describe the ability
    to override the default.

    Attributes:
        env (Environment): The gym environment used to collect transitions and
            train the agent.
        n_envs (int): The number of environments, only > 1 if env is a VectorEnv.
        agent (Optional[Agent], optional): The agent stored within the runner, used
            by the step and run methods. Defaults to None.
        extra_specs (dict, optional): Extra entries to include in the replay buffer.
            Defaults to {}.
        capacity (int, optional): Maximum size of the replay buffer. Defaults to
            1_000_000.
        buffer (Optional[BaseBuffer], optional): __description__. Defaults to None.
        rollout_len (int, optional): Number of environment steps to perform as part
            of the step method. Defaults to 1.
        batch_size (int, optional): Number of transitions to sample from the replay
            buffer per batch. Defaults to 100.
        warmup_len (int, optional): Number of steps to perform with the agent before
            regular rollouts begin. Defaults to 0.
        n_batches (int, optional): Number of batches to sample from the replay buffer
            per runner step. Defaults to 1.
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
        extra_specs: dict = {},
        capacity: int = 1_000_000,
        buffer: Optional[BaseBuffer] = None,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        n_batches: int = 1,
        n_step: int = 1,
        eval_env: Optional[Env] = None,
        gatherer: Optional[Gatherer] = None,
    ) -> None:
        """_summary_

        Args:
            env (Env): The gym environment used to collect transitions and
                train the agent.
            agent (Optional[Agent], optional): The agent stored within the runner, used
                by the step and run methods. Defaults to None.
            extra_specs (dict, optional): Extra entries to include in the replay buffer.
                Defaults to {}.
            capacity (int, optional): Maximum size of the replay buffer. Defaults to
                1_000_000.
            buffer (Optional[BaseBuffer], optional): __description__. Defaults to None.
            rollout_len (int, optional): Number of environment steps to perform as part
                of the step method. Defaults to 1.
            batch_size (int, optional): Number of transitions to sample from the replay
                buffer per batch. Defaults to 100.
            warmup_len (int, optional): Number of steps to perform with the agent before
                regular rollouts begin. Defaults to 10_000.
            n_batches (int, optional): Number of batches to sample from the replay buffer
                per runner step. Defaults to 1.
            n_step (int, optional): Number of environment steps to store within a single
                transition. Defaults to 1.
            eval_env (Optional[Env], optional): An optional separate environment to
                be used for evaluation, must not be a VectorEnv. Defaults to None.
            gatherer (Optional[Gatherer], optional): An optional gatherer to be used by
                the runner. Defaults to None.

        Raises:
            TypeError: Trying to use a VectorEnv with off-policy runner.
        """

        if isinstance(env, VectorEnv):
            raise TypeError("VectorEnv's not yet compatible with off-policy runner")

        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = TreeBuffer(env, capacity, extra_specs, n_step)

        self.capacity = capacity
        self.extra_specs = extra_specs
        self.batch_size = batch_size
        self.n_batches = n_batches
        super().__init__(
            env, agent, rollout_len, warmup_len, n_step, eval_env, gatherer
        )

    def _warm_start(self):
        """Step through environment with freshly initialised agent, to collect
        transitions before training via the agents update method.

        Returns:
            Transition: stacked Transitions from environment
            int: number of Transitions collected
        """
        rollout_transitions, num_transitions = super()._warm_start()
        if num_transitions:
            self.buffer.store(rollout_transitions, num_transitions)

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

        agent = agent if self.agent is None else self.agent
        rollout_transitions, num_transitions = self._rollout(
            self.rollout_len, agent, transform
        )
        if num_transitions:
            self.buffer.store(rollout_transitions, num_transitions)
        k = min(self.batch_size, len(self.buffer))
        if self.n_batches > 1:
            return [self.buffer.sample(k) for _ in range(self.n_batches)]
        else:
            return self.buffer.sample(k)

    def reset(self) -> None:
        """Perform any necessary resets, such as for the replay buffer and
        gatherer.

        TODO: Move resetting of buffer to itself, currently we end up defaulting
        to a tree buffer even if not originally used.
        """
        super().reset()
        del self.buffer
        self.buffer = TreeBuffer(self.env, self.capacity, self.extra_specs)

    def update(self, data: dict):
        """Perform any necessary updates to the replay buffer.

        data (dict): A dictionary containing the indices in the 'idxs'
        key     and the other keys/values to be updated.
        """
        self.buffer.update(data)

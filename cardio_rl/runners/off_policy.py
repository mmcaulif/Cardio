from typing import Callable, Optional

from gymnasium.experimental.vector import VectorEnv

from cardio_rl import Agent, BaseRunner, Gatherer
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Environment, Transition


class OffPolicyRunner(BaseRunner):
    """The runner is the high level orchestrator that deals with the different
    components and data, it contains a gatherer, your agent and any replay buffer
    you might have. The runner calls the gatherer's step function as part its own
    step function, or as part of its built in warmup (for collecting a large amount
    of initial data with your agent) and burnin (for randomly stepping through an
    environment, not collecting data, such as for initialising normalisation values)
    methods. The runner can either be used via its run method (which iteratively
    calls the runner.step and the agent.update methods) or with each mothod individually
    with its step method if you'd like more finegrained control.
    """

    def __init__(
        self,
        env: Environment,
        agent: Optional[Agent] = None,
        extra_specs: dict = {},
        capacity: int = 1_000_000,
        buffer: Optional[TreeBuffer] = None,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        n_batches: int = 1,
        n_step: int = 1,
        eval_env: Optional[Environment] = None,
        gatherer: Optional[Gatherer] = None,
    ) -> None:
        """_summary_

        Args:
            env (Env): _description_
            agent (Optional[Agent], optional): _description_. Defaults to None.
            extra_specs (dict, optional): _description_. Defaults to {}.
            capacity (int, optional): _description_. Defaults to 1_000_000.
            rollout_len (int, optional): _description_. Defaults to 1.
            batch_size (int, optional): _description_. Defaults to 100.
            warmup_len (int, optional): _description_. Defaults to 10_000.
            n_batches (int, optional): _description_. Defaults to 1.
            n_step (int, optional): _description_. Defaults to 1.
            eval_env (Env, optional): _description_. Defaults to None.
            gatherer (Optional[Gatherer], optional): _description_. Defaults to None.

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
        self.n_step = n_step

        super().__init__(
            env, agent, rollout_len, warmup_len, n_step, eval_env, gatherer
        )  # type: ignore

    def _warm_start(self):
        rollout_transitions, num_transitions = super()._warm_start()
        if num_transitions:
            self.buffer.store(rollout_transitions, num_transitions)

    def step(
        self, transform: Optional[Callable] = None, agent: Optional[Agent] = None
    ) -> Transition | list[Transition]:
        """Main method to step through environment with
        agent, to collect transitions, add them to your replay
        buffer and then sample batches from the buffer to pass
        to your agent's update function.

        Args:
            agent (Agent): Can optionally pass a specific agent to
                step through environment with

        Returns:
            batch (list[Transition]): A list of Transitions
                sampled from the replay buffer. Length of batch
                is equal to self.num_batches
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
        """Perform any necessary resets, such as for the replay buffer
        and gatherer.

        TODO: Move resetting of buffer to itself, currently we end up defaulting
        to a tree buffer even if not originally used.
        """
        super().reset()
        del self.buffer
        self.buffer = TreeBuffer(self.env, self.capacity, self.extra_specs)

    def update(self, data: dict):
        """_summary_

        Args:
            data (dict): _description_
        """
        self.buffer.update(data)

from typing import Callable, Optional

from gymnasium import Env

from cardio_rl import Agent, BaseRunner
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition


class OffPolicyRunner(BaseRunner):
    """_summary_"""

    def __init__(
        self,
        env: Env,
        agent: Optional[Agent] = None,
        extra_specs: dict = {},
        capacity: int = 1_000_000,
        buffer: Optional[TreeBuffer] = None,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        n_batches: int = 1,
        n_step: int = 1,
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
        """

        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = TreeBuffer(env, capacity, extra_specs, n_step)

        self.capacity = capacity
        self.extra_specs = extra_specs

        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_step = n_step

        super().__init__(env, agent, rollout_len, warmup_len, n_step)

    def _warm_start(self):
        rollout_transitions, num_transitions = super()._warm_start()
        if num_transitions:
            self.buffer.store(rollout_transitions, num_transitions)

    def step(
        self, transform: Optional[Callable] = None, agent: Optional[Agent] = None
    ) -> list[Transition]:
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
        batch_samples = [self.buffer.sample(k) for _ in range(self.n_batches)]
        return batch_samples

    def reset(self) -> None:
        """Perform any necessary resets, such as for the replay buffer
        and gatherer.
        """
        super().reset()
        del self.buffer
        self.buffer = TreeBuffer(self.env, self.capacity, self.extra_specs)

    def update(self, data: dict):
        """_summary_

        Args:
            data (_type_): _description_
        """
        self.buffer.update(data)

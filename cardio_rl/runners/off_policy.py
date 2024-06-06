from typing import Optional

from gymnasium import Env

from cardio_rl import Agent, BaseRunner
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition


class OffPolicyRunner(BaseRunner):
    def __init__(
        self,
        env: Env,
        agent: Agent,
        extra_specs: dict = {},
        capacity: int = 1_000_000,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        n_batches: int = 1,
        n_step: int = 1,
    ) -> None:
        self.buffer = TreeBuffer(env, capacity, extra_specs, n_step)
        self.capacity = capacity
        self.extra_specs = extra_specs

        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_step = n_step

        super().__init__(env, agent, rollout_len, warmup_len, n_step)

    def _rollout(
        self, steps: int, agent: Optional[Agent]
    ) -> tuple[list[Transition], int]:
        """Internal method to step through environment for a
        provided number of steps. Return the collected
        transitions and how many were collected, then stores
        transitions in the replay buffer.

        Args:
            steps (int): Number of steps to take in environment
            agent (Agent): Can optionally pass a specific agent to
                step through environment with

        Returns:
            rollout_transitions (Transition): stacked Transitions
                from environment
            num_transitions (int): number of Transitions collected
        """
        rollout_transitions, num_transitions = super()._rollout(steps, agent)  # type: ignore
        if num_transitions:
            prepped_batch = self.transform_batch(rollout_transitions)
            self.buffer.store(prepped_batch, num_transitions)
        return rollout_transitions, num_transitions

    def step(self, agent: Optional[Agent] = None) -> list[Transition]:
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
        _, _ = self._rollout(self.rollout_len, agent)
        k = min(self.batch_size, len(self.buffer))
        batch_samples = [self.buffer.sample(k) for _ in range(self.n_batches)]
        return batch_samples

    def reset(self) -> None:
        """Perform any necessary resets, for the replay buffer
        and gatherer
        """
        del self.buffer
        self.buffer = TreeBuffer(self.env, self.capacity, self.extra_specs)
        super().reset()

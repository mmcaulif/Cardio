from typing import Optional
from gymnasium import Env
from cardio_rl.agent import Agent
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.runners import BaseRunner
from cardio_rl import Transition


class OffPolicyRunner(BaseRunner):
    def __init__(
        self,
        env: Env,
        extra_specs: dict = {},
        capacity: int = 1_000_000,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        n_batches: int = 1,
        n_step: int = 1,
        agent: Optional[Agent] = None
    ) -> None:
        self.buffer = TreeBuffer(env, capacity, extra_specs, n_step)
        self.capacity = capacity
        self.extra_specs = extra_specs

        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_step = n_step

        super().__init__(env, rollout_len, warmup_len, n_step, agent)

    def _rollout(self, steps: int, agent: Optional[Agent]) -> tuple[list[Transition], int]:
        rollout_batch, num_transitions = super()._rollout(steps, agent)
        prepped_batch = self.transform_batch(rollout_batch)
        self.buffer.store(prepped_batch, num_transitions)
        return rollout_batch, num_transitions

    def step(self, agent: Optional[Agent] = None) -> list[Transition]:
        agent = agent if self.agent is None else self.agent
        self._rollout(self.rollout_len, agent)
        k = min(self.batch_size, len(self.buffer))
        batch_samples = [self.buffer.sample(k) for _ in range(self.n_batches)]
        return batch_samples

    def reset(self) -> None:
        del self.buffer
        self.buffer = TreeBuffer(self.eval_env, self.capacity, self.extra_specs)
        super().reset()

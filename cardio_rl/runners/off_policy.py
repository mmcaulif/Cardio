from gymnasium import Env
from cardio_rl.agent import Agent
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.gatherer import Gatherer
from cardio_rl.runners import BaseRunner
from cardio_rl import Transition


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
        gatherer: Gatherer = Gatherer(),
        n_batches: int = 1,
    ) -> None:
        self.buffer = TreeBuffer(env, capacity, extra_specs)
        self.capacity = capacity
        self.extra_specs = extra_specs

        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_step = gatherer.n_step

        super().__init__(env, agent, rollout_len, warmup_len, gatherer)

    def _rollout(self, length: int) -> Transition:
        rollout_batch = super()._rollout(length)
        self.buffer.store(self.prep_batch(rollout_batch), length)
        return rollout_batch

    def step(self) -> list[Transition]:
        self._rollout(self.rollout_len)
        k = min(self.batch_size, len(self.buffer))
        batch_samples = [self.buffer.sample(k) for _ in range(self.n_batches)]
        return batch_samples

    def prep_batch(self, batch: Transition) -> Transition:
        """
        takes the batch (which will be a dict) and processes them
        """
        # need to redo after implementing replay buffer class
        return batch

    def reset(self) -> None:
        del self.buffer
        self.buffer = TreeBuffer(self.env, self.capacity, self.extra_specs)
        super().reset()

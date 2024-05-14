import logging
from typing import Optional
from gymnasium import Env
from cardio_rl.agent import Agent
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.gatherer import Gatherer
from cardio_rl.runners import BaseRunner


class OffPolicyRunner(BaseRunner):
    def __init__(
        self,
        env: Env,
        agent: Agent,
        extra_specs: dict = {},
        capacity: Optional[int] = 1_000_000,
        rollout_len: int = 1,
        batch_size: int = 100,
        warmup_len: int = 10_000,
        collector: Gatherer = Gatherer(),
        n_batches: int = 1,
    ) -> None:
        
        self.buffer = TreeBuffer(env, capacity, extra_specs)

        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_step = collector.n_step

        super().__init__(env, agent, warmup_len, collector)   

    def _warm_start(self):
        # Need to figure out whether to perform a random policy or not during warmup
        batch = self.collector.step(self.agent, self.warmup_len)
        for transition in batch:
            self.buffer.store(self.prep_batch(transition))
        
        logging.info('### Warm up finished ###')

    def step(self):
        rollout_batch = self.collector.step(self.agent, self.rollout_len)

        for transition in rollout_batch:
            # Remove for loops when storing multiple transitions
            self.buffer.store(self.prep_batch(transition))

        k = min(self.batch_size, len(self.buffer))

        if self.n_batches == 1:
            batch_samples = self.buffer.sample(k)
        else:
            batch_samples = [self.buffer.sample(k) for _ in range(self.n_batches)]

        return batch_samples

    def prep_batch(self, batch):
        """
        takes the batch (which will be a list of transitons) and processes them to be seperate etc.
        """
        # need to redo after implementing replay buffer class

        if self.n_step == 1:
            return batch

        # elif self.reduce == False:
        #     processed_batch = []
        #     for n_step_transition in batch:
        #         transition = n_step_transition
        #         processed_batch.append([*transition])

        #     return processed_batch
        
        # else:
        #     processed_batch = []
        #     for n_step_transition in batch:
        #         s, a, r, s_p, d, i = n_step_transition
        #         s = s[0]
        #         a = a[0]
        #         r_list = list(r)
        #         s_p = s_p[-1]
        #         d = any(d)
        #         i = i
        #         processed_batch.append([s, a, r_list, s_p, d, i])

        #     return self.transition(*zip(*processed_batch))
        
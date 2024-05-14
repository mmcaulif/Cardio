import logging
from tqdm import trange
from gymnasium import Env
from cardio_rl.gatherer import Gatherer
from cardio_rl.agent import Agent


"""
Refactor to be a base runner that is inherited by both on-policy and off-policy
"""

class BaseRunner:
    def __init__(
        self,
        env: Env,
        agent: Agent,
        warmup_len: int = 1_000,
        collector: Gatherer = Gatherer(),
    ) -> None:

        self.env = env
        self.warmup_len = warmup_len

        self.collector = collector

        self.agent = agent
        self.agent.setup(self.env)   

        self.collector._init_env(self.env)
        self._warm_start()       

    def _rollout(self, length):
        rollout_batch = self.collector.step(self.agent, length)    # Needs to return a single dict with each value!
        return rollout_batch

    def _warm_start(self):
        # Need to figure out whether to perform a random policy or not during warmup
        batch = self._rollout(self.agent, self.warmup_len)
        logging.info('### Warm up finished ###')
        
    def step(self):
        rollout_batch = self._rollout(self.agent, 1)
        batch_samples = self.prep_batch(rollout_batch)
        return batch_samples
    
    def run(self, rollouts: int = 1_000_000):
        for _ in trange(rollouts):
            data = self.step()
            self.agent.update(data)

    def prep_batch(self, batch):
        return batch
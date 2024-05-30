import logging

import numpy as np
import cardio_rl as crl
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
        rollout_len: int = 1,
        warmup_len: int = 1_000,
        gatherer: Gatherer = Gatherer(),
    ) -> None:

        self.env = env
        self.rollout_len = rollout_len
        self.warmup_len = warmup_len

        self.gatherer = gatherer
        self._burn_in(0)

        self.agent = agent
        self.agent.setup(self.env)   

        self.gatherer._init_env(self.env)
        self._warm_start()      

    def _burn_in(self, length: int) -> dict:
        dummy = Agent()
        dummy.setup(self.env)
        self.gatherer.step(dummy, length)

    def _rollout(self, length: int) -> dict:
        rollout_batch = self.gatherer.step(self.agent, length)
        # Below is a temporary measure, move this to the gatherer
        if rollout_batch:
            rollout_batch = crl.tree.stack(rollout_batch)
        return rollout_batch

    def _warm_start(self):
        # Need to figure out whether to perform a random policy or not during warmup
        batch = self._rollout(self.warmup_len)
        logging.info('### Warm up finished ###')
        
    def step(self) -> dict:
        rollout_batch = self._rollout(self.rollout_len)
        batch_samples = self.prep_batch(rollout_batch)
        return batch_samples
    
    def run(self, rollouts: int = 1_000_000, eval_interval: int = 10_000, eval_episodes: int = 10) -> None:
        for i in trange(rollouts):
            data = self.step()
            self.agent.update(data)
            if i % eval_interval == 0:
                self.evaluate(eval_episodes)
    
    def evaluate(self, episodes: int) -> None:
        metrics = {
            'return': np.zeros(episodes),
            'length': np.zeros(episodes)
        }
        for e in range(episodes):
            state, _ = self.env.reset()
            returns = 0
            steps = 0
            while True:
                action, _ = self.agent.step(state)
                next_state, reward, done, trun, _ = self.env.step(action)
                returns += reward
                steps += 1
                state = next_state
                if done or trun:
                    metrics['return'][e] = returns
                    metrics['length'][e] = steps
                    break

    def update_agent(self, new_agent: Agent, setup = False):
        self.agent = new_agent
        if setup:
            self.agent.setup(self.env)  

    def prep_batch(self, batch: dict) -> dict:
        return batch

    def reset(self) -> None:
        self.gatherer.reset()
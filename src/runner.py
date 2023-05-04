from collections import deque
from typing import NamedTuple
from gatherer import Collector
from transitions import TorchTransition
from policies import REGISTRY as pol_REGISTRY
import sys
import random

# https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
# faster replay memory

def get_offpolicy_runner(
        env,
        policy,
        freq,
        capacity,
        batch_size,
        collector=Collector,
        train_after=10000):
    
    return Runner(
        env,
        policy,
        freq,
        False,
        True,
        capacity,
        batch_size,
        collector,
        train_after)

class Runner():
    def __init__(
            self,
            env,
            policy,
            length,
            flush,
            sampler,
            capacity,
            batch_size,
            collector=Collector,
            train_after=None
        ) -> None:

        self.env = env
        self.length = length
        self.flush = flush
        self.sampler = sampler
        self.capacity = capacity

        self.batch_size = batch_size

        self.train_after = train_after

        self.er_buffer = deque(maxlen=self.capacity)

        self.collector = collector(env, 100000)

        if self.train_after:
            self._warm_start(self.train_after)
            print('\n### Warm up finished ###\n')
            
        self.policy = self._set_up_policy(policy)


    def _warm_start(
            self,
            steps
        ):

        batch = self.collector.rollout(None,  pol_REGISTRY['random'](self.env), steps)

        if not self.flush:
            for transition in batch:
                self.er_buffer.append(transition)

        pass

    def _set_up_policy(self, policy):
        if policy == 'argmax':
            return pol_REGISTRY['argmax'](self.env)
        
        elif policy == 'random':
            return pol_REGISTRY['random'](self.env)
        
        elif policy == 'gaussian':
            return pol_REGISTRY['random'](self.env)

        pass

    def get_batch(
            self,
            net,
        ):
        
        self.net = net
        
        batch = self.collector.rollout(net, self.policy, self.length)

        if not self.flush:
            for transition in batch:
                self.er_buffer.append(transition)

            if self.sampler:
                k = min(self.batch_size, len(self.er_buffer))
                batch = random.sample(list(self.er_buffer), k)

            return self.prep_batch(batch)

        return self.prep_batch(batch)

    def prep_batch(
        self,
        batch
        ):

        """
        takes the batch (which will be a list of transitons) and processes them to be seperate etc.
        """

        return TorchTransition(*zip(*batch))    # try return it as dict maybe? or just something fancier and seperated
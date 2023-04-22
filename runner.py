from collections import deque
from typing import NamedTuple
from gatherer import Collector
from transitions import TorchTransition
import sys
import random

# https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
# faster replay memory

def get_offpolicy_runner(
        env,
        freq,
        capacity,
        batch_size,
        collector=Collector,
        train_after=10000):
    
    return Runner(
        env,
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


    def _warm_start(
            self,
            steps
        ):

        batch = self.collector.rollout(None, 'random', steps)

        if not self.flush:
            for transition in batch:
                self.er_buffer.append(transition)

        pass

    def get_batch(
            self,
            net,
            policy,
        ):
        
        self.net = net
        
        batch = self.collector.rollout(net, policy, self.length)

        if not self.flush:
            for transition in batch:
                self.er_buffer.append(transition)

            if self.sampler:
                batch = random.sample(list(self.er_buffer), self.batch_size)

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
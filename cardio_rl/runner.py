from collections import deque
from .gatherer import Collector
from .transitions import REGISTRY as tran_REGISTRY
from .transitions import BaseTransition
from .policies import BasePolicy, REGISTRY as pol_REGISTRY
import random

# https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
# faster replay memory

def get_offpolicy_runner(
        env,
        policy,
        length,
        capacity,
        batch_size,
        n_step=1,
        collector=Collector,
        train_after=10000):
    
    return Runner(
        env,
        policy,
        length,
        True,
        capacity,
        batch_size,
        n_step,
        collector,
        train_after)

def get_onpolicy_runner(
        env,
        policy,
        length,
        collector=Collector,):
    
    return Runner(
        env,
        policy,
        length,
        False,
        None,
        None,
        collector,
        0)

class Runner():
    def __init__(
            self,
            env,
            policy = None,
            sampler = False,
            capacity = None,
            batch_size = None,
            collector = None,
            backend = 'numpy'
        ) -> None:

        """
        Need to implement default arguments
        """

        # Can maybe remove environment as an argument of the runner
        self.env = env

        # Maybe combine sampler and capacity into one argument?
        self.sampler = sampler
        self.capacity = capacity
        self.batch_size = batch_size

        self.er_buffer = deque(maxlen=self.capacity)

        self.collector = collector
        self.rollout_len = collector.rollout_len
        self.warmup_len = collector.warmup_len
        self.n_step = collector.n_step

        self.backend = backend

        self._warm_start()
            
        self.policy = self._set_up_policy(policy)
        self.transition = self._set_up_transition(backend)

    def _warm_start(
            self,
            net=None,
            policy=None,           
        ):

        batch = self.collector.warmup(net, policy)

        if self.sampler:
            for transition in batch:
                self.er_buffer.append(transition)
        
        print('\n### Warm up finished ###\n')
        pass

    def _set_up_policy(self, policy):
        if isinstance(policy, str):
            return pol_REGISTRY[policy](self.env)

        elif isinstance(policy, BasePolicy):
            return policy

        # add warning
        return 
    
    def _set_up_transition(self, backend):
        """
        Maybe change name of backend to transition_type, better discription
        """
        if isinstance(backend, str):
            return tran_REGISTRY[backend]

        # isinstance(A, B) didn't work, this is a temp workaround
        elif backend.__base__ is BaseTransition:
            return backend

        # add warning
        return 

    def get_batch(
            self,
            net,
        ):
        
        self.net = net        
        batch = self.collector.rollout(net, self.policy)

        if self.sampler:
            for transition in batch:
                self.er_buffer.append(transition)

            k = min(self.batch_size, len(self.er_buffer))
            batch = random.sample(list(self.er_buffer), k)

        return self.prep_batch(batch)

    def prep_batch(
        self,
        batch
        ):
        """
        takes the batch (which will be a list of transitons) and processes them to be seperate etc.
        """

        if self.n_step == 1:
            # need to add argument for this (instead of querying the regisry each time)!
            #return tran_REGISTRY[self.backend](*zip(*batch))
            return self.transition(*zip(*batch))
        
        else:
            processed_batch = []
            for n_step_transition in batch:
                # transition = tran_REGISTRY["numpy"](*zip(*n_step_transition))
                transition = self.transition(*zip(*n_step_transition))
                s = transition.s[0]
                a = transition.a[0]
                r_list = list(transition.r)
                s_p = transition.s_p[-1]
                d = any(transition.d)
                i = transition.i
                processed_batch.append([s, a, r_list, s_p, d, i])

            return tran_REGISTRY[self.backend](*zip(*processed_batch))
        
import logging
from gymnasium import Env
from cardio_rl.transitions import REGISTRY as tran_REGISTRY
from cardio_rl.transitions import BaseTransition
from cardio_rl.buffers.circular_buffer import CircErTable
from cardio_rl.policies import BasePolicy, REGISTRY as pol_REGISTRY
from cardio_rl.gatherer import Gatherer
from cardio_rl.module import Module

# https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
# faster replay memory

class Runner(Module):
    def __init__(
            self,
            env: Env,
            policy: BasePolicy = None,
            capacity: int = 1_000_000,
            er_buffer: CircErTable = None,
            batch_size: int = 100,
            collector: Gatherer = Gatherer(),
            n_batches: int = 1,
            reduce: bool = True,
            backend: str = 'numpy',
        ) -> None:

        # Can maybe remove environment as an argument of the runner
        self.env = env

        # Maybe combine sampler and capacity into one argument?
        self.batch_size = batch_size
        self.n_batches = n_batches

        if er_buffer == None and capacity != None:
            transition = self._set_up_transition(backend)
            self.er_buffer = CircErTable(env, capacity, transition)
            self.sampler = True
        elif er_buffer == None and capacity == None:            
            self.sampler = False
        else:
            self.er_buffer = er_buffer
            self.sampler = True

        self.collector = collector
        self.rollout_len = collector.rollout_len
        self.n_step = collector.n_step

        self.reduce = reduce
        self.backend = backend        

        self.policy = self._set_up_policy(policy)
        self.collector._init_env(self.env)
        self._warm_start()            

    def _warm_start(
            self,      
        ):

        batch = self.collector.warmup(pol_REGISTRY['random'](self.env))

        if self.sampler:
            for transition in batch:
                # print(transition)
                self.er_buffer.store(transition)

        # exit()
        
        logging.info('### Warm up finished ###')
        self.collector._init_policy(self.policy)
        pass

    def _set_up_policy(self, policy):
        if isinstance(policy, str):
            return pol_REGISTRY[policy](self.env)

        elif isinstance(policy, BasePolicy):
            return policy

        else:
            # add warning
            return 
    
    def _set_up_transition(self, backend):
        """
        Maybe change name of backend to transition_type, better discription
        """
        if isinstance(backend, str):
            return tran_REGISTRY[backend]

        # isinstance(A, B) didn't work, this is a temp workaround
        # elif backend.__base__ is BaseTransition:
        #     return backend

        # else:
        #     # add warning
        #     return 

    def step(
            self,
            net,
        ):
        
        self.net = net        
        rollout_batch = self.collector(self.rollout_len, net)

        if not self.sampler:            
            return self.prep_batch(rollout_batch)

        else:
            for transition in rollout_batch:
                self.er_buffer.store(transition)

            k = min(self.batch_size, len(self.er_buffer))
            
            batch_samples = []
            for _ in range(self.n_batches):
                batch_samples.append(self.prep_batch(self.er_buffer.sample(k))) 

            return batch_samples


    def prep_batch(
        self,
        batch
        ):
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
        
from collections import deque
from cardio_rl.policies import BasePolicy
from cardio_rl.logger import Logger
import numpy as np
import gymnasium as gym

class Collector():
    def __init__(
            self,
            env,
            rollout_len = 1,
            warmup_len = 0,
            n_step = 1,
            logger_kwargs = None
        ) -> None:        

        # environment init (move to function)
        self.env = env      
        self.rollout_len = rollout_len        

        if self.rollout_len == -1:
            self.ret_if_term = True
            self.rollout_len += 1000000
        else: 
            self.ret_if_term = False   

        self.warmup_len = warmup_len
        self.n_step = n_step      

        if logger_kwargs:
            self.logger = Logger(**logger_kwargs)
        else:
            self.logger = Logger()

        # env initialisation
        self.state, _ = self.env.reset()

    def _env_step(self, policy):
        a = policy(self.state, self.net)
        s_p, r, d, t, info = self.env.step(a)
        self.logger.step(r, d, t)
        #      [self.state, a, r, s_p, d, *info]
        return (self.state, a, r, s_p, d, info), s_p, d, t

    def warmup(            
        self,
        net = None,
        policy = None,
    ):        
        # Maybe move this check to the runner?
        if self.warmup_len == None:
            return list(deque())
        
        if policy == None:
            policy = BasePolicy(self.env)
        
        self.net = net
        gather_buffer = deque()
        step_buffer = deque(maxlen=self.n_step)     

        for _ in range(self.warmup_len):
            transition, next_state, done, trun = self._env_step(policy)
            step_buffer.append(transition)
            if len(step_buffer) == self.n_step:
                if self.n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = next_state
            if done or trun:
                self.state, _ = self.env.reset()
                step_buffer = deque(maxlen=self.n_step)

        return list(gather_buffer)

    def rollout(
        self,
        net,
        policy,
    ):        
        self.net = net
        gather_buffer = deque()
        step_buffer = deque(maxlen=self.n_step)   

        for _ in range(self.rollout_len):
            transition, next_state, done, trun = self._env_step(policy)
            step_buffer.append(transition)
            if len(step_buffer) == self.n_step:
                if self.n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = next_state
            if done or trun:
                self.state, _ = self.env.reset()
                step_buffer = deque(maxlen=self.n_step)
                
                if self.ret_if_term:
                    return list(gather_buffer)

        return list(gather_buffer)


class VectorCollector():
    def __init__(
            self,
            env,
            num_envs = 2,
            rollout_len = 1,
            warmup_len = 0,
            n_step = 1,
        ) -> None:  
        
        # https://gymnasium.farama.org/api/vector/#async-vector-env            
        env_list = [lambda: gym.wrappers.RecordEpisodeStatistics(env) for _ in range(num_envs)]
        # Getting errors when using AsyncVectorEnv
        self.env = gym.vector.SyncVectorEnv(env_list)
        # self.env = gym.wrappers.RecordEpisodeStatistics(env)
        
        self.state, _ = self.env.reset()        

        self.rollout_len = rollout_len        

        if self.rollout_len == -1:
            self.ret_if_term = True
            self.rollout_len += 1000000
        else: 
            self.ret_if_term = False   

        self.warmup_len = warmup_len
        self.n_step = n_step        

        # metrics
        self.episodes = 0
        self.total_steps = 0
        self.ep_rew = 0
        self.epsiode_window = deque(maxlen=50)

    def warmup(            
        self,
        net = None,
        policy = None,
    ):
        
        # Maybe move this check to the runner?
        if self.warmup_len == None:
            return deque()
        
        if policy == None:
            policy = BasePolicy(self.env)
        
        self.net = net
        gather_buffer = deque()
        step_buffer = deque(maxlen=self.n_step)     

        for _ in range(self.warmup_len):
            self.total_steps += 1
            a = policy(self.state, self.net)
            s_p, r, d, t, info = self.env.step(a)

            step_buffer.append([self.state, a, r, s_p, d])
            if len(step_buffer) == self.n_step:

                if self.n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = s_p

        return list(gather_buffer)

    def rollout(
        self,
        net,
        policy,
    ):        
        self.net = net
        gather_buffer = deque()
        step_buffer = deque(maxlen=self.n_step)   

        for _ in range(self.rollout_len):
            self.total_steps += 1
            a = policy(self.state, self.net)

            s_p, r, d, t, info = self.env.step(a)
            
            if info:
                fin = info['final_info'][0]
                if fin:
                    self.ep_rew = 0.1*(fin['episode']['r']) + 0.9*self.ep_rew
                    # print(self.ep_rew)
                    # print(fin['episode']['r'])

            step_buffer.append([self.state, a, r, s_p, d])
            if len(step_buffer) == self.n_step:

                if self.n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = s_p

        return list(gather_buffer)
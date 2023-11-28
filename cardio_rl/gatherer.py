from collections import deque
from gymnasium import Env
from cardio_rl.module import Module
from cardio_rl.logger import Logger

class Gatherer(Module):
    def __init__(
            self,
            rollout_len: int = 1,
            warmup_len: int = 1_000,
            n_step: int = 1,
            take_every: int = 1,
            logger_kwargs = dict(
                    log_interval = 5_000,
                    episode_window=50,
                    tensorboard=False,
                )
        ) -> None:        

        self.rollout_len = rollout_len        

        if self.rollout_len == -1:
            self.ret_if_term = True
            self.rollout_len += 1000000
        else: 
            self.ret_if_term = False   

        self.warmup_len = warmup_len
        self.n_step = n_step   
        self.take_every = take_every 
        self.take_count = 0

        if logger_kwargs:
            self.logger = Logger(**logger_kwargs)
        else:
            self.logger = Logger()

        # env initialisation
        self.step_buffer = deque(maxlen=self.n_step)
    
    def _init_env(self, env: Env):
        self.env = env
        self.state, _ = self.env.reset()

    def _init_policy(self, policy):
        self.policy = policy

    def _env_step(self, policy=None, state=None, net=None):
        a = policy(state, net)
        s_p, r, d, t, info = self.env.step(a)
        self.logger.step(r, d, t)
        d = d or t
        return (self.state, a, r, s_p, d), s_p, d, t

    def warmup(            
        self,
        policy
    ):     
        if self.warmup_len:
            self.policy = policy
            return self.step(self.warmup_len)

        else:
            return []

    def step(
        self,
        length,
        net=None,
    ):
        self.net = net
        gather_buffer = deque()   

        for _ in range(length):
            transition, next_state, done, trun = self._env_step(self.policy, self.state, self.net)
            self.step_buffer.append(transition)
            if len(self.step_buffer) == self.n_step:
                
                if self.take_count % self.take_every == 0:
                    if self.n_step == 1:
                        gather_buffer.append(*list(self.step_buffer))
                    else:
                        gather_buffer.append(list(self.step_buffer))
 
                self.take_count += 1

            self.state = next_state
            if done or trun:
                self.state, _ = self.env.reset()
                self.step_buffer = deque(maxlen=self.n_step)
                self.policy.reset()
                if self.ret_if_term:
                    return list(gather_buffer)

        return list(gather_buffer)


# class VectorCollector():
#     def __init__(
#             self,
#             env,
#             num_envs = 2,
#             rollout_len = 1,
#             warmup_len = 0,
#             n_step = 1,
#             logger_kwargs = None
#         ) -> None:  
        
#         # https://gymnasium.farama.org/api/vector/#async-vector-env            
#         env_list = [lambda: gym.wrappers.RecordEpisodeStatistics(env) for _ in range(num_envs)]
#         # Getting errors when using AsyncVectorEnv
#         self.env = gym.vector.SyncVectorEnv(env_list)
#         # self.env = gym.wrappers.RecordEpisodeStatistics(env)
        
#         self.state, _ = self.env.reset()        

#         self.rollout_len = rollout_len        

#         if self.rollout_len == -1:
#             self.ret_if_term = True
#             self.rollout_len += 1000000
#         else: 
#             self.ret_if_term = False   

#         self.warmup_len = warmup_len
#         self.n_step = n_step        

#         # metrics
#         if logger_kwargs:
#             self.logger = Logger(n_envs=num_envs, **logger_kwargs)
#         else:
#             self.logger = Logger(n_envs=num_envs)

#     def init_policy(self, policy):
#         self.policy = policy

#     def _env_step(self, policy):
#         a = policy(self.state, self.net)
#         s_p, r, d, t, info = self.env.step(a)
#         self.logger.vector_step(r, d, t)
#         d_combined = [False] * len(d)

#         for i in range(len(d_combined)):
#             d_combined[i] = d[i] or t[i]
            
#         return (self.state, a, r, s_p, d_combined, info), s_p, d, t

#     def warmup(            
#         self,
#         net = None,
#         policy = None,
#     ):
        
#         # Maybe move this check to the runner?
#         if self.warmup_len == None:
#             return deque()
        
#         warmup_policy = BasePolicy(self.env)
        
#         self.net = net
#         gather_buffer = deque()
#         step_buffer = deque(maxlen=self.n_step)     

#         for _ in range(self.warmup_len):
#             self.total_steps += 1
#             a = policy(self.state, self.net)
#             s_p, r, d, t, info = self.env.step(a)

#             step_buffer.append([self.state, a, r, s_p, d])
#             if len(step_buffer) == self.n_step:

#                 if self.n_step == 1:
#                     gather_buffer.append(*list(step_buffer))
#                 else:
#                     gather_buffer.append(list(step_buffer))

#             self.state = s_p

#         return list(gather_buffer)

#     def rollout(
#         self,
#         net,
#         policy,
#     ):        
#         self.net = net
#         gather_buffer = deque()
#         step_buffer = deque(maxlen=self.n_step)   

#         for _ in range(self.rollout_len):
#             transition, next_state, _, _ = self._env_step(policy)
#             step_buffer.append(transition)
#             if len(step_buffer) == self.n_step:
#                 if self.n_step == 1:
#                     gather_buffer.append(*list(step_buffer))
#                 else:
#                     gather_buffer.append(list(step_buffer))

#             self.state = next_state

#         return list(gather_buffer)
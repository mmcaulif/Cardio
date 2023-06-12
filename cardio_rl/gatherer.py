from collections import deque
from cardio_rl.policies import BasePolicy
import numpy as np

class Collector():
    def __init__(
            self,
            env,
            rollout_len = 1,
            warmup_len = 0,
            n_step = 1,
        ) -> None:

        self.env = env

        if False:
            # https://gymnasium.farama.org/api/vector/#async-vector-env
            num_envs = 1
            env_list = [lambda: gym.make("Pendulum-v1", g=9.81)] * num_envs
            env = gym.vector.AsyncVectorEnv(env_list)

        self.rollout_len = rollout_len        

        if self.rollout_len == -1:
            self.ret_if_term = True
            self.rollout_len += 1000000
        else: 
            self.ret_if_term = False   

        self.warmup_len = warmup_len
        self.n_step = n_step
        
        # environment init (move to function)
        self.state, _ = env.reset()

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
            if d or t:
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
            self.total_steps += 1
            a = policy(self.state, self.net)
            s_p, r, d, t, info = self.env.step(a)

            # metrics
            self.ep_rew += r

            step_buffer.append([self.state, a, r, s_p, d])
            if len(step_buffer) == self.n_step:

                if self.n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = s_p
            if d or t:
                self.episodes += 1
                self.epsiode_window.append(self.ep_rew)
                self.ep_rew = 0
                self.state, _ = self.env.reset()
                step_buffer = deque(maxlen=self.n_step)

                if self.episodes % 10 == 0:
                    print(f"Average reward after {self.episodes} episodes or {self.total_steps} timesteps: {np.mean(self.epsiode_window)}")

                if self.ret_if_term:
                    return list(gather_buffer)

        return list(gather_buffer)   
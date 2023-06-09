from collections import deque
from src.policies import Base_policy
import torch as th
import numpy as np

class Collector():
    def __init__(
            self,
            env,
            capacity,
        ) -> None:

        self.env = env
        self.capacity = capacity
        
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
        length = 0,
        n_step = 1
    ):
        if policy == None:
            policy = Base_policy(self.env)

        return self.rollout(net, policy, length, n_step)

    def rollout(
        self,
        net,
        policy,
        length,
        n_step
    ):
        
        self.net = net
        gather_buffer = deque()
        step_buffer = deque(maxlen=n_step)

        if length == -1:
            ret_when_term = True
            length += 1000000
        else: 
            ret_when_term = False        

        for _ in range(length):
            self.total_steps += 1
            a = policy(self.state, self.net)
            s_p, r, d, t, info = self.env.step(a)

            # metrics
            self.ep_rew += r

            step_buffer.append([self.state, a, r, s_p, d])
            if len(step_buffer) == n_step:

                if n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = s_p
            if d or t:
                self.episodes += 1
                self.epsiode_window.append(self.ep_rew)
                self.ep_rew = 0
                self.state, _ = self.env.reset()
                step_buffer = deque(maxlen=n_step)

                if self.episodes % 10 == 0:
                    print(f"Average reward after {self.episodes} episodes or {self.total_steps} timesteps: {np.mean(self.epsiode_window)}")

                if ret_when_term:
                    return list(gather_buffer)

        return list(gather_buffer)   
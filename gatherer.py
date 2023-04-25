from collections import deque
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
        self.state, _ = self._reset()

        # metrics
        self.ep_rew = 0

        pass

    def _step(
        self,
        a
    ):
        return self.env.step(a)

    def _reset(
        self
    ):
        return self.env.reset()
    
    def agent_step(
        self,
        policy,
        state=None
    ):
        
        if policy == 'argmax':
            input = th.from_numpy(self.state).float()
            out = self.net(input).detach().numpy()
            return np.argmax(out)

        if policy == 'gaussian':
            input = th.from_numpy(self.state).float()
            out = self.net(input)

            mean, std = out[0].detach().numpy(), out[1].detach().numpy()

            noise = np.random.uniform()

            return (np.tanh(mean + (std * noise)) * self.env.action_space.high)

        return self.env.action_space.sample()

    def rollout(
        self,
        net,
        policy,
        length
    ):
        
        self.net = net

        buffer = deque()

        for _ in range(length):
            a = self.agent_step(policy)
            s_p, r, d, t, info = self._step(a)

            # metrics
            self.ep_rew += r

            buffer.append([self.state, a, r, s_p, d])
            self.state = s_p
            if d or t:
                print(self.ep_rew)
                self.ep_rew = 0
                self.state, _ = self._reset()

        return list(buffer)
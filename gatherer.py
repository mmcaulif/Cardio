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
        self.episodes = 0
        self.total_steps = 0
        self.ep_rew = 0
        self.epsiode_window = deque(maxlen=50)

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
            mean, log_std = self.net(input)
            std = log_std.exp()

            dist = th.distributions.Normal(mean, std)
            a_sampled = th.nn.Tanh()(dist.rsample()).detach()

            return a_sampled.numpy() * self.env.action_space.high + 0

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
            self.total_steps += 1
            a = self.agent_step(policy)
            s_p, r, d, t, info = self._step(a)

            # metrics
            self.ep_rew += r

            buffer.append([self.state, a, r, s_p, d])
            self.state = s_p
            if d or t:
                self.episodes += 1
                self.epsiode_window.append(self.ep_rew)
                self.ep_rew = 0
                self.state, _ = self._reset()

                if self.episodes % 10 == 0:
                    print(f"Average reward after {self.episodes} episodes or {self.total_steps} timesteps: {np.mean(self.epsiode_window)}")

        return list(buffer)
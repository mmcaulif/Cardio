# from .basic_policies import Epsilon_argmax_policy 

import torch as th
import numpy as np
import random

# should make as classes with __call__ functions

class Basepolicy():
    def __init__(self, env):
        self.env = env        
        
    def __call__(self, state, net):
        return self.env.action_space.sample()
    
 
class Epsilon_Deterministic_policy(Basepolicy):
    def __init__(self, env):
        super().__init__(env)
        self.eps = 0.9
        self.noise = True 
        
    def __call__(self, state, net):
        input = th.from_numpy(state).float()

        if np.random.rand() > self.eps:
            self.eps = max(0.05, self.eps*0.99)   
            out = net(input)         

            if self.noise:
                mean = th.zeros_like(out)
                noise = th.normal(mean=mean, std=0.1).clamp(-0.5, 0.5)
                out = (out + noise)
        
            return out.clamp(-1, 1).detach().numpy()

        else:
            self.eps = max(0.1, self.eps*0.999)
            return self.env.action_space.sample()
                 

class Epsilon_argmax_policy(Basepolicy):
    def __init__(self, env, eps = 0.0, min_eps = 0.0, ann_coeff = 0.9):
        super().__init__(env)
        self.eps = eps
        self.min_eps = min_eps
        self.ann_coeff = ann_coeff

    def __call__(self, state, net):
        input = th.from_numpy(state).float()

        if np.random.rand() > self.eps:
            self.eps = max(self.min_eps, self.eps*self.ann_coeff)   
            out = net(input).detach().numpy()
            return np.argmax(out)  

        else:
            self.eps = max(self.min_eps, self.eps*self.ann_coeff)
            return self.env.action_space.sample()
        
    

class Gaussian_policy(Basepolicy):
    def __init__(self, env):
        super().__init__(env)

    def __call__(self, state, net):
        input = th.from_numpy(state).float()
        mean, log_std = net(input)
        std = log_std.exp()

        dist = th.distributions.Normal(mean, std)
        a_sampled = th.nn.Tanh()(dist.rsample()).detach()

        return a_sampled.numpy() * self.env.action_space.high + 0


class Noisy_naf_policy(Basepolicy):
    def __init__(self, env):
        super().__init__(env)

    def __call__(self, state, net):
        input = th.from_numpy(state).float()
        out, _, _, _ = net(input)
        return out.detach().numpy()

class Categorical_policy(Basepolicy):
    def __init__(self, env):
        super().__init__(env)

    def __call__(self, state, net):
        input = th.from_numpy(state).float()
        probs = net(input)
        dist = th.distributions.Categorical(probs)
        return dist.sample().detach().numpy()

REGISTRY = {}

REGISTRY["random"] = Basepolicy
REGISTRY["epsilon_deterministic"] = Epsilon_Deterministic_policy
REGISTRY["epsilon_argmax"] = Epsilon_argmax_policy
REGISTRY["gaussian"] = Gaussian_policy
REGISTRY["naf"] = Noisy_naf_policy
REGISTRY["categorical"] = Categorical_policy
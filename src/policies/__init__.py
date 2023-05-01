# from .basic_policies import Epsilon_argmax_policy 

import torch as th
import numpy as np

# should make as classes with __call__ functions

class Random_policy():
    def __init__(self, env):
        self.env = env        
        
    def __call__(self, state, net):
        return self.env.action_space.sample()
    
class Epsilon_argmax_policy(Random_policy):
    def __init__(self, env):
        super().__init__(env)

    def __call__(self, state, net):
        input = th.from_numpy(state).float()
        out = net(input).detach().numpy()
        return np.argmax(out)

class Epsilon_naf_policy(Random_policy):
    def __init__(self, env):
        super().__init__(env)

    def __call__(self, state, net):
        input = th.from_numpy(state).float()
        _, _, _, out = net(input).detach().numpy()
        return out

REGISTRY = {}

REGISTRY["random"] = Random_policy
REGISTRY["argmax"] = Epsilon_argmax_policy
REGISTRY["naf"] = Epsilon_naf_policy
from typing import NamedTuple
import torch as th
import numpy as np
import jax.numpy as jnp

"""
All transition functions should output each vairable as shape [N, d]
where N is the batch size and d is the dimension of that specific variable,
i.e. d = [N, 1], r = [N, 1], s = [N, obs_dimensions]
"""

"""
Transitions need to be revisited some time later, I got extermely 
confused when trying to use them so it is likely unintuitive, each batch ends up
having variables named s,a,r,s_p,d but you need to call the batch function itself to
convert them to their respective back end? Maybe have the backend converstion as a method
of the Transition class itself? That could be a good fix
"""

class BaseTransition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

    def __call__(self):
        s = self.s
        a = self.a
        r = self.r
        s_p = self.s_p
        d = self.d
        return s, a, r, s_p, d

class TorchTransition(BaseTransition):

    def __call__(self):
        s = th.from_numpy(np.array(self.s)).float()
        a = th.from_numpy(np.array(self.a)).unsqueeze(1).float()
        r = th.from_numpy(np.array(self.r)).unsqueeze(1).float()
        s_p = th.from_numpy(np.array(self.s_p)).float()
        d = th.from_numpy(np.array(self.d)).unsqueeze(1).int()
        return s, a, r, s_p, d
    
class JaxTransition(BaseTransition):

    def __call__(self):
        s = jnp.asarray(np.array(self.s))
        a = jnp.asarray(np.array(self.a))
        r = jnp.expand_dims(jnp.asarray(np.array(self.r)), -1)
        s_p = jnp.asarray(np.array(self.s_p))
        d = jnp.expand_dims(jnp.asarray(np.array(self.d)), -1)
        return s, a, r, s_p, d
    
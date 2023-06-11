from typing import NamedTuple
import torch as th
import numpy as np

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
        a = th.from_numpy(np.array(self.a)).float().unsqueeze(1)
        r = th.FloatTensor(self.r).unsqueeze(1)
        s_p = th.from_numpy(np.array(self.s_p)).float()
        d = th.IntTensor(self.d).unsqueeze(1)
        return s, a, r, s_p, d
    
class JaxTransition(BaseTransition):

    def __call__(self):
        return NotImplementedError

        s = th.from_numpy(np.array(self.s)).float()
        a = th.from_numpy(np.array(self.a)).float().unsqueeze(1)
        r = th.FloatTensor(self.r).unsqueeze(1)
        s_p = th.from_numpy(np.array(self.s_p)).float()
        d = th.IntTensor(self.d).unsqueeze(1)
        return s, a, r, s_p, d
    
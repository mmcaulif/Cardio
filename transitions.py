from typing import NamedTuple
import torch as th
import numpy as np

class Transition(NamedTuple):
    s: list  # state
    a: float  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

class TorchTransition(Transition):

    def __call__(self):

        s = th.from_numpy(np.array(self.s)).float()
        a = th.from_numpy(np.array(self.a)).float().unsqueeze(1)
        r = th.FloatTensor(self.r).unsqueeze(1)
        s_p = th.from_numpy(np.array(self.s_p)).float()
        d = th.IntTensor(self.d).unsqueeze(1)
        return s, a, r, s_p, d
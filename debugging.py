import gymnasium as gym
import numpy as np
import torch as th
from cardio_rl import Runner, Collector
from cardio_rl.transitions import BaseTransition
from typing import NamedTuple
from debug_env import CardioDebugEnv

class TestTransition(BaseTransition):
    def __call__(self):
        s = th.from_numpy(np.array(self.s)).float()
        a = th.from_numpy(np.array(self.a)).unsqueeze(1).float()
        r = th.from_numpy(np.array(self.r)).unsqueeze(1).float()
        s_p = th.from_numpy(np.array(self.s_p)).float()
        d = th.from_numpy(np.array(self.d)).unsqueeze(1).int()
        return s, a, r, s_p, d, self.i

env = CardioDebugEnv()

"""
N-step collection currently works in what may be an unexpected way,
rollout steps are spent filling the n-step buffer, which is then flushed
when the episode is terminated. i.e.
In the DebugEnv, with n_step=4 and a rollout_len=10-13, only 7 transitions 
are collected as rollout steps 0-3 are spent filling the buffer (you actually
an error if rollout_steps < n_step), and then when the episode terminates, steps
10-13 are spent filling it again. 
Need to consider options/possibilities...
"""

runner = Runner(
	env=env,
	policy='random',
    sampler=True,
    capacity=1_000,
    batch_size=2,
	collector=Collector(
		env=env,
		rollout_len=2,
        warmup_len=6,
        n_step=4,
        take_every=2,
		),
	reduce=False,
	backend='pytorch'
)

batch: NamedTuple = runner.get_batch(net=None)

# Needing to call batch as a function is awkward and unnecessary, should try find a workaround
print(len(runner.er_buffer))
# exit()
s, a, *_ = batch()
print(s, a)

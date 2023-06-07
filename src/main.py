import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from policies import Basepolicy
from runner import get_offpolicy_runner

"""
Duelling Double DQN with n-step returns used for verification of runner
"""

class Q_duelling(nn.Module):
	def __init__(self, state_dim, action_dim, dueling):
		super(Q_duelling, self).__init__()
		self.dueling = dueling

		self.l1 = nn.Linear(state_dim, 128)

		self.adv = nn.Sequential(
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, action_dim)
		)

		self.val = nn.Sequential(
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	def forward(self, state):
		q = F.relu(self.l1(state))
		v = self.val(q)
		a = self.adv(q)

		if self.dueling:
			return v + (a - a.mean())

		return a

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

env = gym.make('CartPole-v1')
runner = get_offpolicy_runner(
	env, 
	Epsilon_argmax_policy(env, 0.5, 0.05, 0.9),
	length=4,
	capacity=100000, 
	batch_size=512, 
	train_after=100,
	n_step=3)

net = Q_duelling(4, 2, True)
targ_net = copy.deepcopy(net)
optimizer = th.optim.Adam(net.parameters(), lr=2.3e-3)
gamma = 0.99
target_update = 10

for t in range(10000):
	batch = runner.get_batch(net)
	s, a, r, s_p, d = batch()

	if runner.n_step !=  1:
		r_list = r.squeeze(1)
		r_nstep = th.zeros(len(r))
		for n in reversed(range(runner.n_step)):
			r_nstep += r_list[:,n] * gamma
		r = r_nstep.unsqueeze(-1)

	q = net(s).gather(1, a.long())

	with th.no_grad():
		a_p = net(s_p).argmax(dim=1).unsqueeze(1)		
		q_p = targ_net(s_p).gather(1, a_p.long())		
		y = r + gamma * q_p * (1 - d)

	loss = F.mse_loss(q, y)

	optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
	optimizer.step()

	if t % target_update == 0:        
		targ_net = copy.deepcopy(net)
		
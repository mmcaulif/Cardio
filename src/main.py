import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from runner import get_offpolicy_runner

runner = get_offpolicy_runner(
	gym.make('CartPole-v1'), 
	'argmax',
	freq=256,
	capacity=100000, 
	batch_size=64, 
	train_after=1000)

class Q_duelling(nn.Module):
	def __init__(self, state_dim, action_dim, dueling):
		super(Q_duelling, self).__init__()
		self.dueling = dueling

		self.l1 = nn.Linear(state_dim, 256)

		self.adv = nn.Sequential(
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim)
		)

		self.val = nn.Sequential(
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, state):
		q = F.relu(self.l1(state))
		v = self.val(q)
		a = self.adv(q)

		if self.dueling:
			return v + (a - a.mean())

		return a

net = Q_duelling(4, 2, False)
targ_net = copy.deepcopy(net)
optimizer = th.optim.Adam(net.parameters(), lr=2.3e-3)

for t in range(10000):
	batch = runner.get_batch(net)

	s, a, r, s_p, d = batch()

	q = net(s).gather(1, a.long())

	with th.no_grad():
		q_p = targ_net(s_p).amax(dim = 1).unsqueeze(1)
		y = r + 0.99 * q_p * (1 - d)
		
	loss = F.mse_loss(q, y)

	optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 10)
	optimizer.step()

	if t % 10 == 0:        
		targ_net = copy.deepcopy(net)
		
import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from runner import get_offpolicy_runner

runner = get_offpolicy_runner(
	gym.make('Pendulum-v1'), 
	'random',
	freq=1,
	capacity=200000, 
	batch_size=256, 
	train_after=10000)

class Q_critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_critic, self).__init__()

		self.net1 = nn.Sequential(
			nn.Linear(state_dim+action_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 1))
		
		self.net2 = nn.Sequential(
			nn.Linear(state_dim+action_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 1))

	def forward(self, state, action):
		sa = th.concat([state, action], dim=-1)
		q1, q2 = self.net1(sa), self.net2(sa)
		return q1, q2
	
	def q1_forward(self, state, action):
		sa = th.concat([state, action], dim=-1)
		q1 = self.net1(sa)
		return q1

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 400),
			nn.ReLU(),)
		
		self.mean = nn.Sequential(
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, action_dim)
		)

		self.std = nn.Sequential(
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, action_dim),
			nn.ReLU()
		)
		
	def forward(self, state):
		h = self.net(state)
		return self.mean(h), th.log(self.std(h))

net = Q_critic(3, 1)
policy = Policy(3, 1)
targ_net = copy.deepcopy(net)
c_optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
pi_optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

for t in range(10000):
	batch = runner.get_batch(policy)

	s, a, r, s_p, d = batch()

	q1, q2 = net(s, a.squeeze(dim=-1))

	with th.no_grad():

		mu, std = policy(s_p)
		noise = th.rand(len(std)).unsqueeze(dim=-1)
		a_p = th.tanh(mu + (std * noise))

		q1_p, q2_p = targ_net(s_p, a_p)

		y = r + 0.98 * th.min(q1_p, q2_p) * (1 - d)
		
	loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 10)
	c_optimizer.step()

	if t % 2 == 0:
		mu, std = policy(s_p)
		noise = th.rand(len(std)).unsqueeze(dim=-1)
		a_sampled = th.tanh(mu + (std * noise))

		policy_loss = -net.q1_forward(s, a_sampled).mean()

		pi_optimizer.zero_grad()
		policy_loss.backward()
		th.nn.utils.clip_grad_norm_(policy.parameters(), 10)
		pi_optimizer.step()

	
	for targ_params, params in zip(targ_net.parameters(), net.parameters()):
		targ_params = (targ_params * 1-0.005) + (params*0.005)
	
	"""
	for targ_params, params in zip(targ_net.parameters(), net.parameters()):
		targ_params.data.copy_(0.005*params.data + (1.0-0.005)*targ_params.data)
	"""
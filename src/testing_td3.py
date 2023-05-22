import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from runner import get_offpolicy_runner

env = gym.make('gym_cartpole_continuous:CartPoleContinuous-v1')

runner = get_offpolicy_runner(
	env, 
	'deterministic',
	length=1,
	capacity=1000000, 
	batch_size=100, 
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
	
	def q1(self, state, action):
		sa = th.concat([state, action], dim=-1)
		q1 = self.net1(sa)
		return q1

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, action_dim))
		
	def forward(self, state):
		return self.net(state)

net = Q_critic(4, 1)
targ_net = copy.deepcopy(net)
policy = Policy(4, 1)
targ_policy = copy.deepcopy(policy)
c_optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
pi_optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
# Need to debug, not performing well enough at all, learning collapses

for steps in range(50000):
	batch = runner.get_batch(policy)

	s, a, r, s_p, d = batch()

	with th.no_grad():
		a_p = targ_policy(s_p)

		mean = th.zeros_like(a_p)
		noise = th.normal(mean=mean, std=0.1).clamp(-0.5, 0.5)

		q1_p, q2_p = targ_net(s_p, (a_p + noise).clamp(-1, 1))
		q_p = th.min(q1_p, q2_p)

		y = r + 0.98 * q_p * (1 - d)

	q1, q2 = net(s, a.squeeze(1))
		
	loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
	c_optimizer.step()

	if steps % 2 == 0:

		policy_loss = -net.q1(s, policy(s)).mean()

		pi_optimizer.zero_grad()
		policy_loss.backward()
		th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
		pi_optimizer.step()
	
		for targ_params, params in zip(targ_net.parameters(), net.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

		for targ_params, params in zip(targ_policy.parameters(), policy.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
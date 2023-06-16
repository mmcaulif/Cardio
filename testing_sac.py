import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
import numpy as np
import gymnasium as gym

from cardio_rl import Runner, Collector

env = gym.make('Pendulum-v1')

runner = Runner(
	env=env,
	policy='beta',
	sampler=True,
	capacity=100000,
	batch_size=256,
	collector=Collector(
		env=env,
		warmup_len=100,
	)
)

class Q_critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_critic, self).__init__()

		self.net1 = nn.Sequential(
			nn.Linear(state_dim+action_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1))
		
		self.net2 = nn.Sequential(
			nn.Linear(state_dim+action_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1))

	def forward(self, state, action):
		sa = th.concat([state, action], dim=-1)
		q1, q2 = self.net1(sa), self.net2(sa)
		return q1, q2

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),)
		
		self.alpha = nn.Sequential(
			nn.Linear(256, action_dim),
			nn.Softplus()
		)

		self.beta = nn.Sequential(
			nn.Linear(256, action_dim),
			nn.Softplus()
		)
		
	def forward(self, state):
		h = self.net(state)
		a = self.alpha(h)
		b = self.beta(h)
		return 1+a, 1+b

critic = Q_critic(3, 1)
actor = Policy(3, 1)
targ_critic= copy.deepcopy(critic)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)

ent_coeff = 0.1

# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

for steps in range(20000):
	batch = runner.get_batch(actor)

	s, a, r, s_p, d = batch()

	with th.no_grad():
		# create action scaling wrapper for continuous environments
		alpha, beta = actor(s_p)
		dist_p = Beta(alpha, beta)
		x_t = dist_p.rsample()
		a_p = (x_t*4)-2

		q1_p, q2_p = targ_critic(s_p, a_p)

		log_prob = dist_p.log_prob(x_t)
		log_prob = log_prob.sum(-1, keepdim=True)

		q_p = th.min(q1_p, q2_p) - (ent_coeff * log_prob)

		y = r + 0.99 * q_p * (1 - d)

	q1, q2 = critic(s, a.squeeze(1))
		
	loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()

	if steps % 2 == 0:
		alpha, beta = actor(s_p)
		dist = Beta(alpha, beta)
		x_t = dist.rsample()
		a_sampled = (x_t*4)-2

		log_prob = dist.log_prob(x_t)
		log_prob = log_prob.sum(-1, keepdim=True)

		policy_loss =  -(th.min(*critic(s, a_sampled)) - (ent_coeff * log_prob)).mean()

		a_optimizer.zero_grad()
		policy_loss.backward()
		th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
		a_optimizer.step()		
	
	for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
		targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
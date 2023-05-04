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
	'gaussian',
	freq=1,
	capacity=1000000, 
	batch_size=256, 
	train_after=1000)

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

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),)
		
		self.mean = nn.Sequential(
			nn.Linear(300, action_dim)
		)

		self.std = nn.Sequential(
			nn.Linear(300, action_dim),
			nn.Tanh()
		)
		
	def forward(self, state):
		h = self.net(state)
		mean = self.mean(h)
		log_std = self.std(h)
		log_std = -5 + 0.5 * (2 - -5) * (log_std + 1)  # From SpinUp / Denis Yarats
		return mean, log_std

net = Q_critic(4, 1)
policy = Policy(4, 1)
targ_net = copy.deepcopy(net)
c_optimizer = th.optim.Adam(net.parameters(), lr=2e-3)
pi_optimizer = th.optim.Adam(policy.parameters(), lr=2e-3)

autotune = False

if autotune:
	target_entropy = -th.prod(th.Tensor(env.action_space.shape)).item()
	log_alpha = th.zeros(1, requires_grad=True)
	alpha = log_alpha.exp().item()
	a_optimizer = th.optim.Adam([log_alpha], lr=2e-3)

else:
	alpha = 0.15

# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

for steps in range(50000):
	batch = runner.get_batch(policy)

	s, a, r, s_p, d = batch()

	with th.no_grad():

		mu, log_std = policy(s_p)
		std = log_std.exp()

		dist_p = Normal(mu, std)
		x_t = dist_p.rsample()
		y_t = nn.Tanh()(x_t).detach()
		a_p = y_t * env.action_space.high + 0

		q1_p, q2_p = targ_net(s_p, a_p)

		log_prob = dist_p.log_prob(x_t)
		log_prob -= th.log(th.from_numpy(env.action_space.high) * (1 - y_t.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)

		q_p = th.min(q1_p, q2_p) - (alpha * log_prob)

		y = r + 0.99 * q_p * (1 - d)

	q1, q2 = net(s, a.squeeze(1))
		
	loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
	c_optimizer.step()

	if steps % 2 == 0:
		mu, log_std = policy(s)
		std = log_std.exp()

		dist = Normal(mu, std)
		x_t = dist.rsample()
		y_t = nn.Tanh()(x_t).detach()
		a_sampled = y_t * env.action_space.high + 0

		log_prob = dist.log_prob(x_t)
		log_prob -= th.log(th.from_numpy(env.action_space.high) * (1 - y_t.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)

		policy_loss =  -(th.min(*net(s, a_sampled)) - (alpha * log_prob)).mean()

		pi_optimizer.zero_grad()
		policy_loss.backward()
		th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
		pi_optimizer.step()

		if autotune:
			# Entropy tuning
			with th.no_grad():
				mu, log_std = policy(s)
				x_t = Normal(mu, log_std.exp()).rsample()

				log_prob = dist.log_prob(x_t)
				log_prob -= th.log(th.from_numpy(env.action_space.high) * (1 - y_t.pow(2)) + 1e-6)
				log_prob = log_prob.sum(-1, keepdim=True)
				
			alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

			a_optimizer.zero_grad()
			alpha_loss.backward()
			a_optimizer.step()
			alpha = log_alpha.exp().item()
	
		for targ_params, params in zip(targ_net.parameters(), net.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
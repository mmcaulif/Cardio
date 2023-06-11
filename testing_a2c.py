import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from src import Runner, Collector

class Critic(nn.Module):
	def __init__(self, state_dim):
		super(Critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1))

	def forward(self, state):
		return self.net(state)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim),
			nn.LogSoftmax(dim=-1))

	def forward(self, state):
		return th.exp(self.net(state))

env = gym.vector.make('CartPole-v1')

runner = Runner(
	env=env,
	policy='categorical',
	collector=Collector(
		env=env,
		rollout_len=5,
	)
)

net = Critic(4)
actor = Policy(4, 2)
c_optimizer = th.optim.Adam(net.parameters(), lr=7e-4)
pi_optimizer = th.optim.Adam(actor.parameters(), lr=7e-4)

"""
Need to implement parallel environments and better returns estimation!
"""

for rollout_steps in range(50000):
	batch = runner.get_batch(actor)

	s, a, r, s_p, d = batch()

	# Need to improve returns estimates with future expected value
	values = net(s)
	returns = th.zeros_like(r)
	ret = 0
		
	for i in reversed(range(len(r))):
		ret += 0.99 * r[i]
		ret *= (1 - d[i])
		returns[i] = ret
	
	loss = F.mse_loss(returns.detach(), values)
	
	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
	c_optimizer.step()
	
	adv = (values - returns).detach()
	log_probs = Categorical(actor(s)).log_prob(a.squeeze(-1))

	policy_loss = -(log_probs * adv).mean()

	pi_optimizer.zero_grad()
	policy_loss.backward()
	th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	pi_optimizer.step()

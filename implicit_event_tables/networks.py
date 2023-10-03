import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
		return self.net1(sa), self.net2(sa)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, action_dim),
			nn.Tanh())
		
	def forward(self, state):
		return self.net(state)

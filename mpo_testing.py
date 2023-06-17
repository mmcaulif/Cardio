import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from cardio_rl import Runner
from cardio_rl import Collector

class Q_duelling(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_duelling, self).__init__()
		self.l1 = nn.Linear(state_dim, 128)
		self.a1 = nn.Linear(128, 128)
		self.a2 = nn.Linear(128, action_dim)
		self.v1 = nn.Linear(128, 128)
		self.v2 = nn.Linear(128, 1)

	def forward(self, state):
		q = F.relu(self.l1(state))
		a = F.relu(self.a1(q))
		a = self.a2(a)
		v = F.relu(self.v1(q))
		v = self.v2(v)
		return v + (a - a.mean())
	

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim),
			nn.Softmax(dim=-1))

	def forward(self, state):
		return self.net(state)
	

env = gym.make('CartPole-v1')

runner = Runner(
	env=env,
	policy='categorical',	# Epsilon_argmax_policy(env, 0.5, 0.05, 0.9),
	sampler=True,
	capacity=100000,
	batch_size=256,
	collector=Collector(
		env=env,
		rollout_len=4,
		warmup_len=1000,
	),
	backend='pytorch'
)

critic = Q_duelling(4, 2)
actor = Actor(4, 2)

targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)

optimizer = th.optim.Adam(critic.parameters(), lr=3e-4)
actor_optimizer = th.optim.Adam(actor.parameters(), lr=3e-4)
gamma = 0.99
target_update = 10

epsilon = 0.2
placeholder_policy = F.softmax(th.ones([8, 2]), dim=-1)

temperature = nn.Parameter(th.ones(1)*0.03)
dual_optim = th.optim.Adam([temperature], lr=1e-4)

for t in range(10000):
	batch = runner.get_batch(actor)
	s, a, r, s_p, d = batch()

	q = critic(s).gather(1, a.long())

	with th.no_grad():
		a_p = critic(s_p).argmax(dim=1).unsqueeze(1)		
		q_p = targ_critic(s_p).gather(1, a_p.long())		
		y = r + gamma * q_p * (1 - d)

	loss = F.mse_loss(q, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	### E-step

	"""
	Temperature goes to Nan, causing qij to go to Nan causing policy to also go to Nan
	thus issue is likely in my implementation of temperature! 
	"""

	energies = critic(s).detach()/temperature
	logits = th.log(energies.mean(-1)).mean(-1)
	temp_loss = (temperature * (epsilon + logits))

	dual_optim.zero_grad()
	temp_loss.backward()
	dual_optim.step()

	q_values = critic(s).detach()
	qij = F.softmax(q_values/temperature.detach(), dim=-1)
	print(temperature)

	### M-step
	probs = actor(s)
	# check kl div implementation
	a_loss = th.sum(qij * th.log(probs/qij), dim=-1)
	a_loss = th.mean(a_loss)
	# kl = th.mean(th.sum(probs * th.log(targ_actor(s).detach()), dim=-1))

	loss_p = -(a_loss)	#  + 0.5 * (0.01 - kl))

	actor_optimizer.zero_grad()
	loss_p.backward()
	th.nn.utils.clip_grad_norm_(actor.parameters(), 0.1)
	actor_optimizer.step()
	
	continue
	if t % 1000 == 0:        
		targ_critic = copy.deepcopy(critic)
		targ_actor = copy.deepcopy(actor)	# using the target actor network speeds up the Nan creation
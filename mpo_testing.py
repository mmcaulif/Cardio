import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from cardio_rl import Runner
from cardio_rl import Collector
from main import ImplicitQ

"""
To do:
-implement e-pochs for E and M step, figure out how many
-implement kl-div penalty for actor loss
-implement alpha dual loss
-make sure kl-div is calculated right
-figure out where to use target actor, presume in weight calculation DOUBLE CHECK AND MAKE SURE
"""

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
	capacity=1_000_000,
	batch_size=256,
	collector=Collector(
		env=env,
		rollout_len=4,
		warmup_len=1000,
	),
	backend='pytorch'
)

critic = ImplicitQ(4, 2)	# move to using double duelling dqn
actor = Actor(4, 2)

targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)

lr = 7e-4

optimizer = th.optim.Adam(critic.parameters(), lr=lr)
actor_optimizer = th.optim.Adam(actor.parameters(), lr=lr)
gamma = 0.99
target_update = 10

eps = 0.1

# check init values of lagrangian multipliers
eta = nn.Parameter(th.ones(1)*0.03)
eta_optim = th.optim.Adam([eta], lr=lr)

# check init values of lagrangian multipliers
alpha = nn.Parameter(th.ones(1)*0.03)
alpha_optim = th.optim.Adam([alpha], lr=lr)

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

	for _ in range(5):
		energies = th.exp(critic(s).detach() / eta)

		# logits = actor(s).detach() * energies
		logits = targ_actor(s).detach() * energies

		logits = logits.mean(-1)
		logits = th.log(logits).mean()
		temp_loss = (eta * (eps + logits))

		eta_optim.zero_grad()
		temp_loss.backward()
		eta_optim.step()

	q_values = critic(s).detach()
	qij = F.softmax(q_values/eta.detach(), dim=-1)

	# print(f'non-para: {qij[0].numpy()}, policy: {actor(s)[0].detach().numpy()}, q vals: {critic(s)[0].detach().numpy()}')

	### M-step
	probs = actor(s)
	# check kl div calculation
	a_loss = th.sum(qij * th.log(probs/qij), dim=-1)
	a_loss = th.mean(a_loss)
	# kl = th.mean(th.sum(probs * th.log(targ_actor(s).detach()), dim=-1))

	loss_p = -(a_loss)	#  + 0.5 * (0.01 - kl))

	actor_optimizer.zero_grad()
	loss_p.backward()
	# th.nn.utils.clip_grad_norm_(actor.parameters(), 0.1)
	actor_optimizer.step()
	
	if t % 10 == 0:        
		targ_critic = copy.deepcopy(critic)
		targ_actor = copy.deepcopy(actor)
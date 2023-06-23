import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Collector

"""
To do:
-figure out how many e-pochs for E and M step
-implement alpha dual loss
-look into what hyperparams to use, maybe good time to try optuna
-look into Acme impl. for how to handle target updates, i.e. freq
 of hard updates or value for tau
"""

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim))

	def forward(self, state):
		return self.net(state)

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

critic = Critic(4, 2)	# move to using double duelling dqn
actor = Actor(4, 2)

targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)

lr = 3e-4

critic_optim = th.optim.Adam(critic.parameters(), lr=lr)
actor_optim = th.optim.Adam(actor.parameters(), lr=lr)
gamma = 0.99
target_update = 10

eps = 0.1

# check init values of lagrangian multipliers, also look into the projection etc.
eta = nn.Parameter(th.ones(1))
eta_optim = th.optim.Adam([eta], lr=lr)

alpha = th.tensor(0.2, requires_grad=True)	# normally 0.1
alpha_optim = th.optim.Adam([alpha], lr=lr)

for t in range(10000):
	batch = runner.get_batch(actor)
	s, a, r, s_p, d = batch()

	### Policy evaluation
	q = critic(s).gather(1, a.long())
	with th.no_grad():
		a_p = critic(s_p).argmax(dim=1).unsqueeze(1)		
		q_p = targ_critic(s_p).gather(1, a_p.long())		
		y = r + gamma * q_p * (1 - d)

	loss = F.mse_loss(q, y)

	critic_optim.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
	critic_optim.step()
	
	### E-step
	for _ in range(5):
		energies = th.exp(critic(s).detach() / eta).mean(-1)
		logits = th.log(energies).mean()
		eta_loss = (eta * (eps + logits))

		eta_optim.zero_grad()
		eta_loss.backward()		
		eta_optim.step()
	
	qij = F.softmax(targ_critic(s) / eta, dim=-1).detach()
	prior_probs = targ_actor(s).detach()

	### M-step
	for _ in range(5):
		probs = actor(s)
		
		# check kl div calculation
		loss_p = th.mean(qij * th.log(probs))

		# check dimensions of kl-div!
		kl = th.mean((probs * th.log(probs / prior_probs)).sum(dim=-1))
		total_loss_p = -(loss_p + alpha.detach() * (eps - kl))

		actor_optim.zero_grad()
		total_loss_p.backward()
		nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
		actor_optim.step()

		# alpha currently not set to autotune
		continue
		alpha_loss = alpha * (eps - kl).detach()
		alpha_optim.zero_grad()
		alpha_loss.backward()
		alpha_optim.step()

	# implement soft updates maybe?
	targ_critic = copy.deepcopy(critic)
	targ_actor = copy.deepcopy(actor)

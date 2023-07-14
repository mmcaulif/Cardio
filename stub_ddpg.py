import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from cardio_rl import Runner, Collector

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env = RescaleAction(env, -1, 1)

runner = Runner(
	env=env,
	policy='whitenoise',
	sampler=True,
	capacity=200_000,
	batch_size=256,
	collector=Collector(
		env=env,
		warmup_len=10_000,
		logger_kwargs=dict(
	    	log_dir = env_name,
		)
	),
	backend='pytorch'
)

class Q_critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim+action_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 1))

	def forward(self, state, action):
		sa = th.concat([state, action], dim=-1)
		return self.net(sa)

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

critic = Q_critic(3, 1)
actor = Policy(3, 1)
targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)

for steps in range(100_000):
	batch = runner.get_batch(actor)

	s, a, r, s_p, d, _ = batch()

	with th.no_grad():
		a_p = targ_actor(s_p)
		q_p = targ_critic(s_p, a_p)
		y = r + 0.98 * q_p * (1 - d)

	q = critic(s, a.squeeze(1))
		
	loss = F.mse_loss(q, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()

	a_sampled = actor(s_p)

	policy_loss =  -(critic(s, a_sampled)).mean()

	a_optimizer.zero_grad()
	policy_loss.backward()
	th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	a_optimizer.step()	

	for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
		targ_params.data.copy_(params.data * 0.01 + targ_params.data * (1.0 - 0.01))

	for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
		targ_params.data.copy_(params.data * 0.01 + targ_params.data * (1.0 - 0.01))
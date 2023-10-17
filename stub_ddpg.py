import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from cardio_rl import Runner, Collector

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

runner = Runner(
	env=env,
	policy='whitenoise',
	capacity=1_000_000,
	collector=Collector(
		warmup_len=100,
		logger_kwargs=dict(log_interval=1_000)
	),
	backend='pytorch'
)

class Q_critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim+action_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1))

	def forward(self, state, action):
		sa = th.concat([state, action], dim=-1)
		return self.net(sa)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim),
			nn.Tanh())
		
	def forward(self, state):
		return self.net(state)

# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
critic = Q_critic(3, 1)
actor = Policy(3, 1)
targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)

for steps in range(300_000):
	s, a, r, s_p, d, _ = runner.get_batch(actor)

	with th.no_grad():
		a_p = targ_actor(s_p)

		mean = th.zeros_like(a_p)
		noise = th.normal(mean=mean, std=0.2).clamp(-0.5, 0.5)
		
		a_p = (a_p + noise).clamp(-1, 1)

		q_p = targ_critic(s_p, a_p)
		y = r + 0.99 * q_p * (1 - d)

	q = critic(s, a.squeeze(1))
		
	loss = F.mse_loss(q, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()

	policy_loss =  -(critic(s, actor(s_p))).mean()

	a_optimizer.zero_grad()
	policy_loss.backward()
	th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	a_optimizer.step()	

	for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
		targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

	for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
		targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from cardio_rl import Runner, Collector
from cardio_rl.buffers.iet_buffer import IeTable


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

env_name = 'BipedalWalker-v3'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

runner = Runner(
	env=env,
	policy='whitenoise',
	sampler=True,
	capacity=200_000,
	batch_size=100,	# batch_size needs to exceed value of k_events
	collector=Collector(
		env=env,
		warmup_len=10_000,
		logger_kwargs=dict(
			log_interval = 5_000,
			episode_window=50,
            tensorboard=True,
			log_dir = 'implicit_event_tables/tb_logs/' + env_name,
		    exp_name = f'vanilla_td3'
		)
	),
)

# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
critic = Q_critic(observation_dim, action_dim)
actor = Policy(observation_dim, action_dim)
targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)

for steps in range(590_000):
	s, a, r, s_p, d, _ = runner.get_batch(actor)

	with th.no_grad():
		a_p = targ_actor(s_p)

		mean = th.zeros_like(a_p)
		noise = th.normal(mean=mean, std=0.2).clamp(-0.5, 0.5)	# target policy noise value in SB3
		
		a_p = (a_p + noise).clamp(-1, 1)

		q_p = th.min(*targ_critic(s_p, a_p))
		y = r + 0.98 * q_p * (1 - d)

	q1, q2 = critic(s, a.squeeze(1))
		
	loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()

	if steps % 2 == 0:
		sa = th.concat([s_p, actor(s_p)], dim=-1)
		policy_loss =  -(critic.net1(sa).mean())	# critic net1 as in sb3

		a_optimizer.zero_grad()
		policy_loss.backward()
		th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
		a_optimizer.step()	

		for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

		for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
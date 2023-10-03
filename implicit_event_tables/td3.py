import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from networks import Q_critic, Policy
from iet_buffer import IeTable

"""
SB3 zoo hyperparams:
  n_timesteps: !!float 3e5
  policy: 'MlpPolicy'
  gamma: 0.98
  buffer_size: 200000
  learning_starts: 10000
  noise_type: 'normal'
  noise_std: 0.1
  gradient_steps: -1
  train_freq: [1, "episode"]
  learning_rate: !!float 1e-3
  policy_kwargs: "dict(net_arch=[400, 300])"
"""

env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

critic = Q_critic(8, 2)
actor = Policy(8, 2)
targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)

buffer = IeTable(k_events=4, cluster_interval=100)

s_t, _ = env.reset()
avg_rew = -999
running_reward = 0
episodes = 0

"""
Warmup loop
"""
for _ in range(10_000):
	a_t = env.action_space.sample()
	s_tp1, r_t, d_t, t_t, _ = env.step(a_t)
	transition = s_t, a_t, r_t, s_tp1, d_t
	buffer.all_encoded_transitions.append(s_t)
	s_t = s_tp1
	if d_t or t_t:
		s_t, _ = env.reset()

s_t, _ = env.reset()

for t in range(100_000):
	"""
	Environment loop
	"""
	if t > 10_000:
		a_t = actor(th.from_numpy(s_t).float())
		mean = th.zeros_like(a_t)
		noise = th.normal(mean=mean, std=0.1).clamp(-0.5, 0.5)
		a_t = (a_t + noise).clamp(-1, 1).detach().numpy()
	else:
		a_t = env.action_space.sample()	

	s_tp1, r_t, d_t, t_t, _ = env.step(a_t)
	running_reward += r_t

	transition = s_t, a_t, r_t, s_tp1, d_t

	"""
	To do: Implement encoding and representation learning
	-https://github.com/sfujim/td7
	"""
	# z_s = False
	# z_sa = (False, z_s)

	buffer.store(transition)

	s_t = s_tp1
	if d_t or t_t:
		episodes += 1
		if avg_rew == -999:
			avg_rew = running_reward
		else:
			avg_rew  = (avg_rew * 0.9) + (running_reward * 0.1)

		s_t, _ = env.reset()
		if episodes % 5 == 0:
			print(f'{episodes}, {t+1}: {avg_rew}')
		running_reward = 0
	

	"""
	Training loop
	"""
	if t > 10_000:
		s, a, r, s_p, d = buffer.sample(100)

		with th.no_grad():
			a_p = targ_actor(s_p)

			mean = th.zeros_like(a_p)
			noise = th.normal(mean=mean, std=0.1).clamp(-0.5, 0.5)
			
			a_p = (a_p + noise).clamp(-1, 1)

			q_p = th.min(*targ_critic(s_p, a_p))
			y = r + 0.98 * q_p * (1 - d)

		q1, q2 = critic(s, a.squeeze(1))
			
		loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

		c_optimizer.zero_grad()
		loss.backward()
		th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
		c_optimizer.step()

		if t % 2 == 0:
			policy_loss =  -(th.min(*critic(s_p, actor(s_p))).mean())

			a_optimizer.zero_grad()
			policy_loss.backward()
			th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
			a_optimizer.step()	

			for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
				targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

			for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
				targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

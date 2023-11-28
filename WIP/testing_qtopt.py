import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from cardio_rl import Runner, Gatherer
from cardio_rl.policies import BasePolicy


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

class CEM():
	''' 
	cross-entropy method, as optimization of the action policy 
	'''
	def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
		self.theta_dim = theta_dim
		self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

	def initialize(self, ini_mean_scale=0.0, ini_std_scale=1.0):
		self.mean = ini_mean_scale*th.ones(self.theta_dim)
		self.std = ini_std_scale*th.ones(self.theta_dim)

	def sample(self, n):
		theta_list = self.mean + th.normal(mean=th.zeros([n, self.theta_dim]), std=th.zeros([n, self.theta_dim])) * self.std
		theta_list = th.clip(theta_list, -1.0, 1.0)
		return theta_list

	def update(self, selected_samples):
		self.mean = th.mean(selected_samples, dim=0)
		# print('mean: ', self.mean)
		self.std = th.std(selected_samples, dim=0)  # plus the entropy offset, or else easily get 0 std
		# print('std: ', self.std)
		# return self.mean, self.std

# https://github.com/quantumiracle/QT_Opt/blob/master/qt_opt_v3.py
class CemPolicy(BasePolicy):
	def __init__(self, env, recurrent=False, hidden_dims=0):
		super().__init__(env, recurrent, hidden_dims)
		self.cem = CEM(2)
		self.cem_update_iter = 2
		self.num_samples = 64
		self.select_num = 6

	def __call__(self, state, net):
		state = th.FloatTensor(state)
		return self.cem_optimal_action(state, net)
	
	def cem_optimal_action(self, state, net):
		states = state.expand(self.num_samples, -1)
		self.cem.initialize() # every time use a new cem, cem is only for deriving the argmax_a'
		for _ in range(self.cem_update_iter):
			actions = self.cem.sample(self.num_samples)
			q_values = net(states, actions).detach().cpu().reshape(-1) # 2 dim to 1 dim
			# max_idx=q_values.argsort()[-1]  # select one maximal q
			idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
			selected_actions = actions[idx]
			self.cem.update(selected_actions)
	
		optimal_action = actions[idx[-1]]
		return optimal_action.numpy()

runner = Runner(
	env=env,
	policy=CemPolicy(env),
	sampler=True,
	capacity=1_000_000,
	batch_size=100,
	collector=Gatherer(
		env=env,
		warmup_len=10_000,
		logger_kwargs=dict(
			log_dir = env_name,
			log_interval = 2_500
		)
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

# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
critic = Q_critic(8, 2)
targ_critic = copy.deepcopy(critic)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)

for steps in range(150_000):
	s, a, r, s_p, d, _ = runner.step(critic)

	with th.no_grad():
		"""
		Convert this to parallel instead of using list comprehension
		"""
		a_p = np.array([runner.policy.cem_optimal_action(next_state, targ_critic) for next_state in s_p])
		a_p = th.from_numpy(a_p).float()
		# exit(a_p)

		mean = th.zeros_like(a_p)
		noise = th.normal(mean=mean, std=0.2).clamp(-0.5, 0.5)
		
		a_p = (a_p + noise).clamp(-1, 1)

		q_p = targ_critic(s_p, a_p)
		y = r + 0.98 * q_p * (1 - d)

	q = critic(s, a.squeeze(1))
		
	loss = F.mse_loss(q, y)

	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()

	for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
		targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

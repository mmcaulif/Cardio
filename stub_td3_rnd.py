import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction
from cardio_rl.policies import BasePolicy
from cardio_rl import Runner, Collector

class RndWhitenoiseDeterministic(BasePolicy):
	def __init__(self, env: gym.Env, obs_dims, output_dims):
		super().__init__(env)
		# architecture can maybe be improved on
		self.rnd_net = nn.Sequential(
			nn.Linear(obs_dims, 32),
			nn.Tanh(),
			nn.Linear(32, 32),
			nn.Tanh(),
			nn.Linear(32, output_dims),
		)

		self.targ_net = nn.Sequential(
			nn.Linear(obs_dims, 32),
			nn.Tanh(),
			nn.Linear(32, 32),
			nn.Tanh(),
			nn.Linear(32, output_dims),
		)
		
	def __call__(self, state, net):
		input = th.from_numpy(state).float()  
		out = net(input)         
		mean = th.zeros_like(out)
		noise = th.normal(mean=mean, std=0.1)   # .clamp(-0.5, 0.5) # unsure if necessary... need to check other implementations
		out = (out + noise)    
		return out.clamp(-1, 1).detach().numpy()
	
	def predictor(self, next_state):
		next_state = th.from_numpy(next_state).float()
		pred = self.rnd_net(next_state)
		targ = self.targ_net(next_state)
		intrinsic_r = th.pow(th.mean(pred-targ, dim=-1, keepdim=True), 2).detach().item()
		return intrinsic_r

	def update(self, rnd_net):
		self.rnd_net = rnd_net

class RndCollector(Collector):
	def _env_step(self, policy=None, warmup=False):
		if warmup:
			a = self.env.action_space.sample()
		else:
			a = policy(self.state, self.net)
		s_p, r, d, t, info = self.env.step(a)
		self.logger.step(r, d, t)
		d = d or t

		r = r + (0.5 * policy.predictor(s_p))

		return (self.state, a, r, s_p, d, info), s_p, d, t

# env_name = 'MountainCarContinuous-v0'
# env_name = 'Pendulum-v1'
env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

runner = Runner(
	env=env,
	policy=RndWhitenoiseDeterministic(env, 8, 1),
	sampler=True,
	capacity=200_000,
	batch_size=100,
	collector=RndCollector(
		env=env,
		warmup_len=10_000,
		logger_kwargs=dict(
			log_interval = 5_000,
			episode_window=40,
			tensorboard=True,
            log_dir='run',
			exp_name = env_name + '_RND'
		)
	),
	backend='pytorch'
)

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

# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
critic = Q_critic(8, 2)
actor = Policy(8, 2)
targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)
rnd_optimizer = th.optim.Adam(runner.policy.rnd_net.parameters(), lr=1e-3)

for steps in range(150_000):
	batch = runner.get_batch(actor)

	s, a, r, s_p, d, _ = batch()

	pred = runner.policy.rnd_net(s_p)
	targ = runner.policy.targ_net(s_p).detach()

	rnd_loss = F.mse_loss(pred, targ)

	rnd_optimizer.zero_grad()
	rnd_loss.backward()
	rnd_optimizer.step()

	with th.no_grad():
		a_p = targ_actor(s_p)

		noise = th.normal(mean=th.zeros_like(a_p), std=0.2).clamp(-0.5, 0.5)		
		a_p = (a_p + noise).clamp(-1, 1)

		q_p = th.min(*targ_critic(s_p, a_p))
		y = r + 0.98 * q_p * (1 - d)

	q1, q2 = critic(s, a.squeeze(1))
		
	loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

	c_optimizer.zero_grad()
	loss.backward()
	c_optimizer.step()

	if steps % 2 == 0:
		policy_loss = -th.min(*critic(s_p, actor(s_p))).mean()

		a_optimizer.zero_grad()
		policy_loss.backward()
		a_optimizer.step()	

		for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

		for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
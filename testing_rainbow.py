import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Collector
from cardio_rl.rainbow_naf.noisy_linear import NoisyLinear

"""
Duelling Double DQN with n-step returns used for verification of runner functionality
-double check NoisyNet implementation
-should maybe implement distributional q-learning (qr or iqn) and PER to fully make rainbow
-PER will be interesting, need to figure out how to do it with current design structure
"""

class Q_duelling(nn.Module):
	def __init__(self, state_dim, action_dim, dueling):
		super(Q_duelling, self).__init__()
		self.dueling = dueling

		self.l1 = NoisyLinear(state_dim, 128)

		self.a1 = NoisyLinear(128, 128)
		self.a2 = NoisyLinear(128, action_dim)

		self.v1 = NoisyLinear(128, 128)
		self.v2 = NoisyLinear(128, 1)

	def forward(self, state):
		q = F.relu(self.l1(state))

		a = F.relu(self.a1(q))
		a = self.a2(a)

		v = F.relu(self.v1(q))
		v = self.v2(v)

		if self.dueling:
			return v + (a - a.mean())

		return a   
	
	def reset_noise(self):
		"""Reset all noisy layers."""
		self.l1.reset_noise()
		self.a1.reset_noise()
		self.a2.reset_noise()
		self.v1.reset_noise()
		self.v2.reset_noise()

env = gym.make('CartPole-v1')

runner = Runner(
	env=env,
	policy='argmax',	# Epsilon_argmax_policy(env, 0.5, 0.05, 0.9),
	sampler=True,
	capacity=100000,
	batch_size=64,
	collector=Collector(
		env=env,
		rollout_len=4,
		warmup_len=1000,
		n_step=3
	),
	backend='pytorch'
)

critic = Q_duelling(4, 2, True)
targ_net = copy.deepcopy(critic)
optimizer = th.optim.Adam(critic.parameters(), lr=2.3e-3)
gamma = 0.99
target_update = 10

for t in range(5000):
	batch = runner.get_batch(critic)
	s, a, r, s_p, d, _ = batch()

	if runner.n_step !=  1:
		r_nstep = th.zeros([len(r), 1])
		for n in reversed(range(runner.n_step)):
			r_nstep += r[:,:,n] * gamma

		r = r_nstep

	q = critic(s).gather(1, a.long())

	with th.no_grad():
		a_p = critic(s_p).argmax(dim=1).unsqueeze(1)		
		q_p = targ_net(s_p).gather(1, a_p.long())		
		y = r + pow(gamma, runner.n_step) * q_p * (1 - d)

	loss = F.mse_loss(q, y)

	optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
	optimizer.step()
	
	critic.reset_noise()
	targ_net.reset_noise()

	if t % target_update == 0:        
		targ_net = copy.deepcopy(critic)
		
import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Collector
from cardio_rl.policies import Epsilon_argmax_policy

"""
A simple duelling DDQN implementation!
"""

class Q_duelling(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_duelling, self).__init__()
		self.l1 = nn.Linear(state_dim, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, action_dim)

	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q
	

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
	),
	backend='pytorch'
)

critic = Q_duelling(4, 2)
targ_net: nn.Module = copy.deepcopy(critic)
optimizer = th.optim.Adam(critic.parameters(), lr=2.3e-3)
gamma = 0.99
target_update = 10

for t in range(5000):
	batch = runner.get_batch(critic)
	s, a, r, s_p, d = batch()

	with th.no_grad():
		# a_p = critic(s_p).argmax(dim=1).unsqueeze(1)		
		# q_p = targ_net(s_p).gather(1, a_p.long())

		q_p = th.max(targ_net(s_p), keepdim=True, dim=-1).values
		# print(q_p, q_p.shape, r.shape, d.shape)

		y = r + gamma * q_p * (1 - d)

	q = critic(s).gather(1, a.long())

	loss = F.mse_loss(q, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if t % target_update == 0:        
		targ_net = copy.deepcopy(critic)
		
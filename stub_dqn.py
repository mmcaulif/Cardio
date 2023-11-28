import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import envpool
from cardio_rl import Runner, Gatherer

env = envpool.make_gymnasium('CartPole-v1')

runner = Runner(
	env=env,
	policy='argmax',
	batch_size=256,
	collector=Gatherer(
		rollout_len=4,
		warmup_len=10_000,
		logger_kwargs=dict(log_interval=5_000)
	),
	backend='pytorch'
)

class Q_critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim)
		)

	def forward(self, state):
		q = self.net(state)
		return q

critic = Q_critic(4, 2)
targ_critic = Q_critic(4, 2)
targ_critic.load_state_dict(critic.state_dict())
optimizer = th.optim.Adam(critic.parameters(), lr=1e-3)

for steps in range(300_000):
	batches = runner(critic)

	for batch in batches:

		s, a, r, s_p, d = batch

		q = critic(s).gather(1, a.long())

		with th.no_grad():
			a_p = critic(s_p).argmax(dim=1, keepdim=True)
			q_p = targ_critic(s_p).gather(1, a_p.long())
			y = r + 0.99 * q_p * (1 - d)

		loss = F.mse_loss(q, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if steps % 250 == 0:
			targ_critic.load_state_dict(critic.state_dict())

import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Collector
from cardio_rl.policies import EpsilonArgmax


class QuantileregressionQ(nn.Module):
	def __init__(self, state_dim, action_dim, quantiles=32):
		super(QuantileregressionQ, self).__init__()
		self.action_dim = action_dim
		self.quantiles = quantiles

		self.net = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, action_dim*quantiles)
		)

	def forward(self, state, reduce=True):
		q = self.net(state).view(-1, self.action_dim, self.quantiles)

		if reduce:
			q = th.mean(q, dim=-1)

		return q
	
def main():
	env = gym.make('CartPole-v1')

	runner = Runner(
		env=env,
		policy=EpsilonArgmax(env, 0.9, 0.05, 0.9),
		sampler=True,
		capacity=100_000,
		batch_size=256,
		collector=Collector(
			env=env,
			rollout_len=4,
			warmup_len=500,
		),
		backend='pytorch'
	)

	critic = QuantileregressionQ(4, 2, 32)
	targ_net: nn.Module = copy.deepcopy(critic)
	optimizer = th.optim.Adam(critic.parameters(), lr=2.3e-3)
	gamma = 0.99
	target_update = 300

	tau = th.FloatTensor([(n/32) for n in range(1, 32+1)])

	for t in range(10000):
		batch = runner.get_batch(critic)
		s, a, r, s_p, d, _ = batch()

		with th.no_grad():
			a_p = critic(s_p).argmax(dim=-1).view(-1, 1, 1)
			q_p = targ_net(s_p, False).gather(1, a_p.expand(256, 1, 32).long())
			y = r.view(-1, 1, 1) + gamma * q_p * (1 - d.view(-1, 1, 1))

		q = critic(s, False).gather(1, a.unsqueeze(-1).expand(256, 1, 32).long())
		
		kappa = 1.0
		u = y - q

		huber_loss = th.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
		qh_loss = abs(tau - (u.detach() < 0).float()) * huber_loss
		loss = qh_loss.sum(-1).mean()

		optimizer.zero_grad()
		loss.backward()
		th.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
		optimizer.step()

		if t % target_update == 0:        
			targ_net = copy.deepcopy(critic)
			
if __name__ == '__main__':
	main()
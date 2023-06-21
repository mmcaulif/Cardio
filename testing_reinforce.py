from cardio_rl import Runner, Collector
import torch as th
import gymnasium as gym
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
		
		self.net = nn.Sequential(
			nn.Linear(4, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 2),
			nn.Softmax(dim=-1)
		)

	def forward(self, state):
		return self.net(state)

env = gym.make('CartPole-v1')

runner = Runner(
	env=env,
	policy='categorical',
	collector=Collector(
		env=env,
		rollout_len=-1,
		),
	backend='pytorch'
)

actor = Actor()
optimizer = th.optim.Adam(actor.parameters(), lr=2e-3)
baseline = 50

for _ in range(3_000):
	batch = runner.get_batch(actor)
	s, a, r, s_p, d = batch()

	running_r = 0
	returns = th.zeros(len(r))
	for i, r_val in enumerate(r):
		running_r *= 0.99
		running_r += r_val
		returns[i] = running_r
	
	returns = reversed(returns)

	probs = actor(s)
	dist = Categorical(probs)
	loss = -th.mean(dist.log_prob(a.squeeze(-1)) * (returns-baseline))

	optimizer.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	optimizer.step()

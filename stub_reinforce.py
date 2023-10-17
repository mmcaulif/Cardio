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

for _ in range(1_000):
	batch = runner.get_batch(actor)
	s, a, r, s_p, d, _ = batch()

	returns = th.zeros_like(r)
	running_r = 0
	for t in reversed(range(len(r))):
		running_r = r[t] + 0.99 * running_r
		returns[t] = running_r

	returns = (returns - returns.mean()) / returns.std()

	probs = actor(s)
	dist = Categorical(probs)
	policy_loss = -th.mean(dist.log_prob(a.squeeze(-1)) * returns.squeeze(-1))

	optimizer.zero_grad()
	policy_loss.backward()
	nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	optimizer.step()	

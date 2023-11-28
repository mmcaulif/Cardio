from cardio_rl import Runner, VectorCollector
import torch as th
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(4, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1))

	def forward(self, state):
		return self.net(state)

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
	collector=VectorCollector(
		env=env,
		num_envs=4,
		rollout_len=15,
		logger_kwargs=dict(
			log_interval=1_000,
			episode_window=500
		)
	),
	backend='pytorch'
)

actor = Actor()
critic = Critic()
a_optimizer = th.optim.RMSprop(actor.parameters(), lr=0.0007, eps=1e-5)
c_optimizer = th.optim.RMSprop(critic.parameters(), lr=0.0007, eps=1e-5)

for _ in range(100_000):
	batch = runner.step(actor)
	s, a, r, s_p, d, _ = batch()

	r = r.squeeze(-2)
	a = a.squeeze(-2)
	d = d.squeeze(-2)

	returns = th.zeros_like(r)
	running_r = critic(s_p[-1]).squeeze(-1).detach()
	for t in reversed(range(len(r))):
		running_r = r[t] + 0.99 * running_r * (1 - d[t])
		returns[t] = running_r

	values = critic(s).squeeze()
	critic_loss = F.mse_loss(returns, values)

	c_optimizer.zero_grad()
	critic_loss.backward()
	nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()

	adv: th.Tensor = (returns - values).detach()

	# adv = ((adv - adv.mean()) / (adv.std() + 1e-5)).detach()

	probs = actor(s)
	dist = Categorical(probs)
	policy_loss = -th.mean(dist.log_prob(a) * adv)

	a_optimizer.zero_grad()
	policy_loss.backward()
	nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	a_optimizer.step()
	continue

	# Below is for Advantage Weighted Regression (need to explore)
	# https://github.com/jcwleo/awr-pytorch/tree/master
	for _ in range(10):
		probs = actor(s)
		dist = Categorical(probs)
		policy_loss = -th.mean(dist.log_prob(a.squeeze(-1)) * th.exp(norm_adv/1.0))

		a_optimizer.zero_grad()
		policy_loss.backward()
		# nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
		a_optimizer.step()

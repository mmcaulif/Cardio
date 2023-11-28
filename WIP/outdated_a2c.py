import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from cardio_rl import Runner, VectorCollector

class Model(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Model, self).__init__()

		self.critic = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1))

		self.actor = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim),
			nn.Softmax(dim=-1))

	def forward(self, state):
		return self.actor(state)

env = gym.make('CartPole-v1')

runner = Runner(
	env=env,
	policy='categorical',
	collector=VectorCollector(
		env=env,
		num_envs=8,
		rollout_len=5,
	)
)

net = Model(4, 2)
optimizer = th.optim.RMSprop(net.parameters(), lr=7e-4, eps=1e-5)

"""
Need to implement parallel environments
-Implement timestep-based logging!
-Debug and benchmark, doesn't seem like it works rn 
-Maybe merge actor and critic into one nn.Module

https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char04%20A2C/A2C.py
"""

timesteps = 0

for rollout_steps in range(10_000):

	timesteps += 5*2

	batch = runner.step(net)

	s, a, r, s_p, d = batch()

	values = net.critic(s)
	returns = th.zeros_like(values)

	ret = th.zeros_like(r)
	running_r = net.critic(s_p[-1]).squeeze(-1).detach()

	for t in reversed(range(len(r))):
		running_r = r[t] + 0.99 * running_r * (1 - d[t])
		ret[t] = running_r

	critic_loss = F.mse_loss(returns, values)
	
	adv = (returns - values).squeeze(-1).detach()

	dist = Categorical(net(s))	
	log_probs = dist.log_prob(a.squeeze(1))
	ent = dist.entropy().sum(-1).mean()

	policy_loss = -(log_probs * adv.detach()).mean()

	if timesteps % 2000 == 0:
		# Check if the trnopy loss recorded in sb3 is before or after being multiplied by the 
		# entropy coefficient
		print(f'{timesteps} = Policy loss: {policy_loss}, Critic loss: {critic_loss}, Entropy loss: {ent}')

	loss = policy_loss +  (0.5 * critic_loss) # - (0.001 * ent) 

	optimizer.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(net.parameters(), 0.5)
	optimizer.step()

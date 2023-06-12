import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from cardio_rl import Runner, VectorCollector

class Critic(nn.Module):
	def __init__(self, state_dim):
		super(Critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1))

	def forward(self, state):
		return self.net(state)

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim),
			nn.LogSoftmax(dim=-1))

	def forward(self, state):
		return th.exp(self.net(state))

env = gym.make('CartPole-v1')

runner = Runner(
	env=env,
	policy='categorical',
	collector=VectorCollector(
		env=env,
		num_envs=4,
		rollout_len=5,
	)
)

critic = Critic(4)
actor = Policy(4, 2)
c_optimizer = th.optim.Adam(critic.parameters(), lr=7e-4)
pi_optimizer = th.optim.Adam(actor.parameters(), lr=7e-4)

"""
Need to implement parallel environments
-Implement timestep-based logging!
-Maybe merge actor and critic into one nn.Module
-debug and inspect returns estimation!
"""

for rollout_steps in range(50000):
	batch = runner.get_batch(actor)

	s, a, r, s_p, d = batch()

	# Need to improve returns estimates with future expected value
	values = critic(s)
	returns = th.zeros_like(r)
		
	for t in reversed(range(len(r))):
		if t == len(r) - 1:			
			ret = r[t]
			ret += 0.99 * critic(s_p[-1,]).T	# have shape issues without transposing
			# can maybe remove the below two lines and move them outside the if statement
			ret *= (1 - d[t])
			returns[t] = ret

		else:
			ret = r[t]
			ret += 0.99 * returns[t+1]
			ret *= (1 - d[t])
			returns[t] = ret

	values = values.flatten()
	returns = returns.flatten()

	print(returns)
	import sys
	sys.exit()
	
	loss = F.mse_loss(returns.detach(), values)
	
	c_optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	c_optimizer.step()
	
	adv = (values - returns).detach()

	print(adv.shape)

	# I think dimensions are working for the value updates, 
	# now just need to fix for the policy gradient updates

	log_probs = Categorical(actor(s))
	
	log_probs = log_probs.log_prob(a.squeeze(-1))

	policy_loss = -(log_probs * adv).mean()

	pi_optimizer.zero_grad()
	policy_loss.backward()
	th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
	pi_optimizer.step()

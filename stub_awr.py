import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Collector
from cardio_rl.policies import EpsilonArgmax

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(4, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 2))

	def forward(self, state):
		return self.net(state)

class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()		
		self.net = nn.Sequential(
			nn.Linear(4, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
			nn.Softmax(dim=-1)
		)

	def forward(self, state):
		return self.net(state)


def main():
	env = gym.make('CartPole-v1')

	runner = Runner(
		env=env,
		policy='categorical',
		sampler=True,
		capacity=100_000,
		batch_size=256,
		collector=Collector(
			env=env,
			rollout_len=16,
			warmup_len=10_000,
		),
		backend='pytorch'
	)

	actor = Actor()
	targ_actor: nn.Module = copy.deepcopy(actor)

	critic = Critic()
	targ_critic: nn.Module = copy.deepcopy(critic)

	# use the default a2c optimiser
	a_optimizer = th.optim.RMSProp(actor.parameters(), lr=5e-4)
	c_optimizer = th.optim.RMSProp(critic.parameters(), lr=1e-3)
	
	gamma = 0.99
	target_update = 10

	for t in range(10000):
		batch = runner.get_batch(actor)
		s, a, r, s_p, d, _ = batch()

		for _ in range(20):
			with th.no_grad():
				q_p = th.max(targ_critic(s_p), keepdim=True, dim=-1).values
				y = r + gamma * q_p * (1 - d)

			q = critic(s).gather(1, a.long())

			c_loss = F.mse_loss(q, y)

			c_optimizer.zero_grad()
			c_loss.backward()
			# nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
			c_optimizer.step()


		for _ in range(100):
			with th.no_grad():
				q_adv = critic(s).gather(1, a.long())

				# a_sampled = th.distributions.Categorical(actor(s)).sample()				
				# v_adv = critic(s).gather(1, a_sampled.unsqueeze(-1).long())

				# below is what the q-net suggests, not what the policy net suggests 
				# should fix
				v_adv = th.max(critic(s), keepdim=True, dim=-1).values
				adv = q_adv - v_adv

				weight = adv.exp()
				weight = th.clamp_max(weight, 20)
			
			probs = actor(s)
			dist = th.distributions.Categorical(probs)
			log_probs = dist.log_prob(a.squeeze(-1))
			policy_loss = -th.mean(log_probs * weight.squeeze(-1))

			a_optimizer.zero_grad()
			policy_loss.backward()
			nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
			a_optimizer.step()


		for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))

		for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
			targ_params.data.copy_(params.data * 0.005 + targ_params.data * (1.0 - 0.005))
			
if __name__ == '__main__':
	main()
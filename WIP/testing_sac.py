import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from cardio_rl import Runner, Gatherer

# env_name = 'LunarLanderContinuous-v2'
env_name = 'BipedalWalker-v3'
# env_name = 'Pendulum-v1'
env = gym.make(env_name)
env = RescaleAction(env, 0, 1)

"""
Soft-actor critic with beta policy

# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
"""

runner = Runner(
	env=env,
	policy='beta',
	sampler=True,
	capacity=300_000,
	batch_size=256,
	n_batches=64,
	collector=Gatherer(
		env=env,
		rollout_len=64,
		warmup_len=10_000,
		logger_kwargs=dict(
            tensorboard=True,
	    	log_dir=env_name,
		    log_interval=5_000,
		    exp_name = 'vanilla_selection_awr_test_0.05_policy_decay'	# _no_WD'
		)
	),
	backend='pytorch'
)

class DroQ(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(DroQ, self).__init__()

		# self.net = nn.Sequential(
		# 	nn.Linear(state_dim+action_dim, 400),
		# 	nn.Dropout(0.01),
		# 	nn.LayerNorm(400),
		# 	nn.ReLU(),
		# 	nn.Linear(400, 300),
		# 	# nn.Dropout(0.01),
		# 	# nn.LayerNorm(300),
		# 	# nn.ReLU(),
		# 	# nn.Linear(300, 300),
		# 	nn.Dropout(0.01),
		# 	nn.LayerNorm(300),
		# 	nn.ReLU(),
		# 	nn.Linear(300, 1))
		
		self.net = nn.Sequential(
			nn.Linear(state_dim+action_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 1)
		)

	def forward(self, sa):
		return self.net(sa)

class Q_critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Q_critic, self).__init__()

		self.net1 = DroQ(state_dim, action_dim)
		self.net2 = DroQ(state_dim, action_dim)

	def forward(self, state, action):
		sa = th.concat([state, action], dim=-1)
		q1, q2 = self.net1(sa), self.net2(sa)
		return q1, q2

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),)
		
		self.alpha = nn.Sequential(
			nn.Linear(300, action_dim),
			nn.Softplus()
		)

		self.beta = nn.Sequential(
			nn.Linear(300, action_dim),
			nn.Softplus()
		)
		
	def forward(self, state):
		h = self.net(state)
		a = self.alpha(h)
		b = self.beta(h)
		return 1+a, 1+b

critic = Q_critic(24, 4)
actor = Policy(24, 4)
targ_critic= copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(critic.parameters(), lr=7.3e-4)	#, weight_decay=2.4e-4)	# weight_decay value from Relu to the rescue paper
a_optimizer = th.optim.Adam(actor.parameters(), lr=7.3e-4, weight_decay=2.4e-4)	# weight_decay value from Relu to the rescue paper

# cleanrl optimises the log of alpha and computes alpha through it each iteration
#^can apply this to AWAC dual variables?
# also check out what cleanrl implements for atari in SAC: https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details_1
log_alpha = nn.Parameter(th.tensor([1.0], requires_grad=True))
ent_coeff = log_alpha.exp().item()	# nn.Parameter(th.tensor([0.2], requires_grad=True))
ent_optim = th.optim.Adam([log_alpha], lr=7.3e-4)	# check if lr is same as critic and actor
H = -4
tau = 0.02

for steps in range(4_532):
	n_batches = runner.step(targ_actor)	# using target actor, prevents policy-churn
	# print(len(n_batches))
	# exit()
	for batch in n_batches:

		s, a, r, s_p, d, _ = batch()

		for _ in range(1):
			with th.no_grad():
				# use target actor net here?
				alpha, beta = actor(s_p)
				dist_p = Beta(alpha, beta)
				a_p = dist_p.rsample()

				q1_p, q2_p = targ_critic(s_p, a_p)

				log_prob = dist_p.log_prob(a_p)
				log_prob = log_prob.sum(-1, keepdim=True)

				q_p = th.min(q1_p, q2_p) - (ent_coeff * log_prob)

				y = r + 0.98 * q_p * (1 - d)

			q1, q2 = critic(s, a.squeeze(1))
				
			loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

			c_optimizer.zero_grad()
			loss.backward()
			th.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
			c_optimizer.step()

		if steps % 2 == 0:
			alpha, beta = actor(s_p)
			dist = Beta(alpha, beta)
			a_sampled = dist.rsample()

			log_prob = dist.log_prob(a_sampled)
			log_prob = log_prob.sum(-1, keepdim=True)

			# AWR-like policy updates for contraining current policy to behavioural 
			with th.no_grad():
				# Need to implement ADV-max like in CRR
				q_adv = th.min(*critic(s, a.squeeze(1)))
				actions = dist.sample((10,)).transpose(0,1)
				# print(a_sampled.shape, actions.shape)
				j_samples = critic(s.unsqueeze(1).expand(-1, 10, -1), actions)[0]
				v_adv = th.max(j_samples, dim=1).values

				# v_adv = th.min(*critic(s, a_sampled))
				adv = q_adv - v_adv
				# weight = adv.exp()
				# weight = th.clamp_max(weight, 20)

				# Binary
				weight = (adv>0.0).float()

			awr_loss = (dist.log_prob(a.squeeze(1)).sum(-1, keepdim=True) * weight) * 0.05
			
			policy_loss =  -(th.min(*critic(s, a_sampled)) - (ent_coeff * log_prob) + awr_loss).mean()

			a_optimizer.zero_grad()
			policy_loss.backward()
			th.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
			a_optimizer.step()		

			ent_loss = -log_alpha.exp() * (log_prob.detach() + H).mean()
			ent_optim.zero_grad()
			ent_loss.backward()
			ent_optim.step()	

			ent_coeff = log_alpha.exp().item()

			# update targets only when actor is updated or not? TD3 paper only updates targets during policy updates
			# in contrast, CleanRL's SAC does so every update step (could be a matter of performance vs stability)
		for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
			targ_params.data.copy_(params.data * tau + targ_params.data * (1.0 - tau))

		for targ_params, params in zip(targ_critic.parameters(), critic.parameters()):
			targ_params.data.copy_(params.data * tau + targ_params.data * (1.0 - tau))

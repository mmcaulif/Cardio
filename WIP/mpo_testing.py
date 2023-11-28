import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Gatherer

"""
To do:
-figure out how many e-pochs to do for E and M step
-look into what hyperparams to use, maybe good time to try optuna
-look into Acme impl. 
Resources:
-https://github.com/ethanluoyc/magi/pull/59
-https://github.com/vwxyzjn/cleanrl/pull/392
-https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
-https://github.com/deepmind/acme/tree/master/acme/agents/jax/mpo
"""

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim))

	def forward(self, state):
		return self.net(state)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim),
			nn.Softmax(dim=-1))

	def forward(self, state):
		return self.net(state)

env = gym.make('LunarLander-v2')

runner = Runner(
	env=env,
	policy='categorical',
	sampler=True,
	capacity=50_000,	# 1_000_000,
	batch_size=256,
	collector=Gatherer(
		env=env,
		rollout_len=4,
		warmup_len=1_000,
		logger_kwargs=dict(
			log_interval=5000,
		)
	),
	backend='pytorch'
)

critic = Critic(8, 4)
actor = Actor(8, 4)

targ_critic = copy.deepcopy(critic)
targ_actor = copy.deepcopy(actor)

# taken from acme
lr = 1e-3
dual_lr = 1e-3

critic_optim = th.optim.Adam(critic.parameters(), lr=lr)
actor_optim = th.optim.Adam(actor.parameters(), lr=lr)
gamma = 0.99

eps = 0.1
init_logs = 0.0
def from_log(x): 
	# projection suggested in paper and acme impl.
	# return F.softplus(th.clamp(x, min=-18)) + 1e-8
	return x.exp()

# check init values of lagrangian multipliers, acme impl. uses clamping to restrict the min value
# update: apparently acme uses 1.0 as init values? these are very unstable when testing...
log_eta = th.tensor(init_logs, requires_grad=True)
eta_optim = th.optim.Adam([log_eta], lr=dual_lr)
eta = from_log(log_eta).item()

log_alpha = th.tensor(init_logs, requires_grad=True)
alpha_optim = th.optim.Adam([log_alpha], lr=dual_lr)
alpha = from_log(log_alpha).item()

for t in range(12500):
	batch = runner.step(actor)
	s, a, r, s_p, d, _ = batch()

	### Policy evaluation
	with th.no_grad():
		# ddqn
		a_p = critic(s_p).argmax(dim=1).unsqueeze(1)		
		q_p = targ_critic(s_p).gather(1, a_p.long())		
		y = r + gamma * q_p * (1 - d)

	q = critic(s).gather(1, a.long())

	loss = F.mse_loss(q, y)

	critic_optim.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
	critic_optim.step()
	
	### E-step
	for _ in range(5):
		energies = th.exp(critic(s).detach() / F.softplus(log_eta)).mean(-1)
		logits = th.log(energies).mean()
		eta_loss = (from_log(log_eta) * (eps + logits))

		eta_optim.zero_grad()
		eta_loss.backward()		
		nn.utils.clip_grad_norm_([log_eta], 40.0)	
		eta_optim.step()

	eta = from_log(log_eta).item()
	
	qij = F.softmax(targ_critic(s) / eta, dim=-1).detach()
	prior_probs = targ_actor(s).detach()

	### M-step
	for _ in range(5):
		probs = actor(s)
		
		# check kl div calculation
		loss_p = th.mean(qij * th.log(probs))

		# check dimensions of kl-div!
		kl = th.mean((probs * th.log(probs / prior_probs)).sum(dim=-1))
		total_loss_p = -(loss_p + alpha * (eps - kl))

		actor_optim.zero_grad()
		total_loss_p.backward()
		nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
		actor_optim.step()

		# is alpha updated inside or outside of M-steps?
		alpha_loss = from_log(log_alpha) * (eps - kl).detach()
		alpha_optim.zero_grad()
		alpha_loss.backward()
		nn.utils.clip_grad_norm_([log_alpha], 40.0)
		alpha_optim.step()
		alpha = from_log(log_alpha).item()

	# implement soft/delayed updates maybe?
	if t % 100 == 0:	# 100 is used in acme impl.
		targ_critic = copy.deepcopy(critic)
		targ_actor = copy.deepcopy(actor)

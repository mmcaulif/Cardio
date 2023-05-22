import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from runner import get_offpolicy_runner
from rainbow_naf.noisy_linear import NoisyLinear

"""
Rainbow components:
1. Dueling: done as part of naf
2. Double DQN: done, greatly improves performance
3. Noisy dqn: done, but needs to be tuned and tried on more complex environments
4. Prioritised replay: should be easy to re-implement
5. Distributional: might be difficult given the nature of the Q-network's architecture
6. N-step learning:
"""

# env = gym.make('gym_cartpole_continuous:CartPoleContinuous-v1')
env = gym.make('LunarLanderContinuous-v2')

runner = get_offpolicy_runner(
	env,
	'naf', 
	freq=1,
	capacity=200000, 
	batch_size=128, 
	train_after=1000)

class NAF(nn.Module):
	"""
	source:
	https://github.com/BY571/Normalized-Advantage-Function-NAF-
	"""
	def __init__(self, state_size, action_size, layer_size):
		super(NAF, self).__init__()
		self.input_shape = state_size
		self.action_size = action_size
				
		self.head_1 = NoisyLinear(self.input_shape, layer_size)
		# self.bn1 = nn.BatchNorm1d(layer_size)
		self.ff_1 = NoisyLinear(layer_size, layer_size)
		# self.bn2 = nn.BatchNorm1d(layer_size)
		self.action_values = nn.Linear(layer_size, action_size)
		self.value = nn.Linear(layer_size, 1)
		self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))     
	
	def reset_noise(self):
		"""Reset all noisy layers."""
		self.head_1.reset_noise()
		self.ff_1.reset_noise()
		# self.action_values.reset_noise()
		# self.value.reset_noise()
		# self.matrix_entries.reset_noise() 
	
	def forward(self, input_, action=None):
		x = th.relu(self.head_1(input_))
		# x = self.bn1(x)
		x = th.relu(self.ff_1(x))
		# x = self.bn2(x)
		action_value = th.tanh(self.action_values(x))
		entries = th.tanh(self.matrix_entries(x))
		V = self.value(x)
		
		action_value = action_value.unsqueeze(-1)
		
		# create lower-triangular matrix
		L = th.zeros((input_.shape[0], self.action_size, self.action_size)).to(input_.device)

		# get lower triagular indices
		tril_indices = th.tril_indices(row=self.action_size, col=self.action_size, offset=0)  

		# fill matrix with entries
		L[:, tril_indices[0], tril_indices[1]] = entries
		L.diagonal(dim1=1,dim2=2).exp_()

		# calculate state-dependent, positive-definite square matrix
		P = L*L.transpose(2, 1)

		"""
		Currently doesnt work with action spaces greater than size 1, need to fix!!
		"""
		
		Q = None
		if action is not None: 
			# assert action.shape == [N, 1 ,1]

			# action = action.transpose(2, 1)
			action_value = action_value.transpose(2, 1)
			print(action.shape, action_value.shape)

			# calculate Advantage:
			# A = (-0.5 * th.matmul(th.matmul((action - action_value).transpose(2, 1), P), (action - action_value))).squeeze(-1)
			A = (-0.5 * th.matmul(th.matmul((action - action_value).transpose(2, 1), P), (action - action_value))).squeeze(-1)

			print(A.shape, V.shape)	# problem is A
			Q = A + V   
		
		P = P.mean(0)   # Somewhere along the calculation P's size gets messes up, need to look into   

		# print(action_value.shape, entries.shape, V.shape)
		dist = th.distributions.MultivariateNormal(action_value.squeeze(-1), th.inverse(P))
		action = dist.sample()
		action = th.clamp(action, min=-1, max=1)
		return action, Q, V, action_value

net = NAF(state_size=8, action_size=2, layer_size=128)
targ_net = copy.deepcopy(net)
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)

tau = 5e-4

for t in range(50000):
	batch = runner.get_batch(net)

	s, a, r, s_p, d = batch()

	_, q, _, _ = net(s, a)

	with th.no_grad():
		a_p, _, _, _ = targ_net(s_p)	# (1)
		
		# _, q_p, v_p, _ = targ_net(s_p, a_p.unsqueeze(-1))	# regular target q-net target computation

		_, q_p, v_p, _ = net(s_p, a_p.unsqueeze(-1))	# double dqn like target computation, works way better

		y = r + 0.99 * q_p * (1 - d)	# (1)
		# y = r + 0.99 * v_p * (1 - d)

		# (1) this was my own additional tweak, need to experiment with further but it appears to peform better!

	loss = F.mse_loss(q, y)

	optimizer.zero_grad()
	loss.backward()
	th.nn.utils.clip_grad_norm_(net.parameters(), 1.)
	optimizer.step()
	
	net.reset_noise()
	targ_net.reset_noise()

	for targ_params, params in zip(targ_net.parameters(), net.parameters()):
		targ_params.data.copy_(params.data * tau + targ_params.data * (1.0 - tau))
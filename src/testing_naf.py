import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from src.runner import get_offpolicy_runner

runner = get_offpolicy_runner(
	gym.make('Pendulum-v1'), 
	freq=1,
	capacity=200000, 
	batch_size=256, 
	train_after=10000)

class NAF(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed):
        super(NAF, self).__init__()
        self.seed = th.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
                
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))
        

    
    def forward(self, input_, action=None):
        x = th.relu(self.head_1(input_))
        x = self.bn1(x)
        x = th.relu(self.ff_1(x))
        x = self.bn2(x)
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
        
        Q = None
        if action is not None:  

            # calculate Advantage:
            A = (-0.5 * th.matmul(th.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V   
        
        # add noise to action mu:
        dist = th.distributions.MultivariateNormal(action_value.squeeze(-1), th.inverse(P))
        action = dist.sample()
        action = th.clamp(action, min=-1, max=1)
        return action, Q, V, action_value

net = NAF(3, 1)
targ_net = copy.deepcopy(net)
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)

for t in range(10000):
	batch = runner.get_batch(net, 'epsilon_naf')

	s, a, r, s_p, d = batch()
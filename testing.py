import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from runner import get_offpolicy_runner

env = gym.make('CartPole-v1')

runner = get_offpolicy_runner(
    env, 
    freq=256,
    capacity=100000, 
    batch_size=64, 
    train_after=1000)

class Critic(nn.Module):
	def __init__(self, state_dim, action_dims):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dims)

	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q

net = Critic(4, 2)
targ_net = copy.deepcopy(net)
optimizer = th.optim.Adam(net.parameters(), lr=2.3e-3)
policy = 'argmax'

for t in range(10000):
    batch = runner.get_batch(net, policy)

    s, a, r, s_p, d = batch()

    q = net(s)
    
    q = q.gather(1, a.long())

    with th.no_grad():
        q_p = targ_net(s_p).amax(dim = 1).unsqueeze(1)
        y = r + 0.99 * q_p * (1 - d)
        
    loss = F.mse_loss(q, y)

    optimizer.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(net.parameters(), 10)
    optimizer.step()

    if t % 20 == 0:        
        targ_net = copy.deepcopy(net)
        
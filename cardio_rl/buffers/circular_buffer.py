import random
import numpy as np
from gymnasium import spaces
from cardio_rl.transitions import TorchTransition


class CircErTable:
	def __init__(self, env, capacity=1_000_000, transition_func=TorchTransition):

		obs_space = env.observation_space
		obs_dims = obs_space.shape

		act_space = env.action_space

		if isinstance(act_space, spaces.Box):
			act_dim = int(np.prod(act_space.shape))
		elif isinstance(act_space, spaces.Discrete):
			act_dim = 1

		self.pos = 0
		self.capacity = capacity
		self.full = False
		self.states =  np.zeros((self.capacity, *obs_dims), dtype=obs_space.dtype)
		self.actions = np.zeros((self.capacity, act_dim,), dtype=act_space.dtype)
		self.rewards = np.zeros((self.capacity,))
		self.next_states = np.zeros((self.capacity, *obs_dims), dtype=obs_space.dtype)
		self.dones = np.zeros((self.capacity))
		self.transition_func = transition_func
	
	def __len__(self):
		"""
		:return: The current size of the buffer
		"""
		if self.full:
			return self.capacity
		return self.pos

	def store(self, transition):
		state, action, reward, next_state, done = transition
		self.states[self.pos] = np.array(state)
		self.actions[self.pos] = np.array(action)
		self.rewards[self.pos] = np.array(reward)
		self.next_states[self.pos] = np.array(next_state)
		self.dones[self.pos] = np.array(done)
			
		self.pos += 1
		if self.pos == self.capacity:
			self.full = True
			self.pos = 0

	def sample(self, batch_size):
		batch_inds = random.sample(range(self.__len__()), int(min(batch_size, self.__len__())))

		states = self.states[batch_inds]
		actions = self.actions[batch_inds]		
		rewards = self.rewards[batch_inds]
		next_states = self.next_states[batch_inds]
		dones = self.dones[batch_inds]
		
		batch = self.transition_func(states, actions, rewards, next_states, dones)
		return batch()

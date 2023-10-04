import random
from cardio_rl.transitions import TorchTransition
from collections import deque


class ErTable:
	def __init__(self, capacity=1_000_000, transition_func=TorchTransition):
		self.table = deque(maxlen=capacity)
		self.transition_func = transition_func
	
	def __len__(self):
		return len(self.table)

	def store(self, transition):
		self.table.append(transition)

	def sample(self, batch_size):
		batch = random.sample(list(self.table), batch_size)
		batch = self.transition_func(*zip(*batch))
		return batch()

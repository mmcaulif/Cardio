import random
from transitions import TorchTransition
from collections import deque


class ErTable:
	def __init__(self, capacity=1_000_000):
		self.table = deque(maxlen=capacity)

	def store(self, transition):
		self.table.append(transition)

	def sample(self, batch_size):
		batch = random.sample(list(self.table), batch_size)
		batch = TorchTransition(*zip(*batch))
		s, a, r, s_p, d = batch()			
		return s, a, r, s_p, d

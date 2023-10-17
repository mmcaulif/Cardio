import random
from cardio_rl.transitions import TorchTransition
from collections import deque


class ErTable:
	def __init__(self, capacity=1_000_000, transition_func=TorchTransition):
		self.buffer = deque(maxlen=capacity)
		self.transition_func = transition_func
	
	def __len__(self):
		return len(self.buffer)

	def store(self, transition):
		self.buffer.append(transition)

	def sample(self, batch_size):
		batch = random.sample(list(self.buffer), int(min(batch_size, self.__len__())))
		batch = self.transition_func(*zip(*batch))
		return batch()

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class QNetBase(nn.Module):
	"""Vanilla Q network base class"""
	def __init__(self, encoder, encoder_dims, hidden_dims, action_dim):
		super(QNetBase, self).__init__()

		self.net = nn.Sequential(
			*encoder,
			nn.Linear(encoder_dims, hidden_dims),
			nn.ReLU(),
			nn.Linear(hidden_dims, action_dim)
		)

	def forward(self, state):
		q = self.net(state)
		return q
	
class QNetConv(QNetBase):
	"""Taken from DQN example in original MinAtar paper"""
	def __init__(self, in_channels, hidden_dims, action_dim):
		
		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

		feature_extractor = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
			nn.Flatten(),
			nn.ReLU(),
		)
		
		QNetBase.__init__(self, feature_extractor, num_linear_units, hidden_dims, action_dim)

class QNetMLP(QNetBase):
	"""Vanilla DQN MLP"""
	def __init__(self, obs_dim, hidden_dims, action_dim):

		feature_extractor = nn.Sequential(
			nn.Linear(obs_dim, hidden_dims),
			nn.ReLU(),
		)
		
		QNetBase.__init__(self, feature_extractor, hidden_dims, hidden_dims, action_dim)
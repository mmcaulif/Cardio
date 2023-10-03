import torch as th
from sklearn.cluster import KMeans
from er_buffer import ErTable


class IeTable:
	def __init__(self, capacity=1_000_000, k_events=4, cluster_interval=10):
		self.tables = [ErTable(capacity) for _ in range(k_events)]
		self.k_events = k_events
		self.all_encoded_transitions = []
		self.encoder = lambda x: x
		self.cluster_interval = cluster_interval
		self.t = 0

	def store(self, transition):
		
		if self.t % self.cluster_interval == 0:
			self.kmeans = KMeans(n_clusters=self.k_events, n_init="auto").fit(self.all_encoded_transitions)

		state, *_ = transition
		encoded_transition = self.encoder(state)
		self.all_encoded_transitions.append(encoded_transition)
		cluster = self.kmeans.predict([encoded_transition])

		cluster = cluster.squeeze(0)
		self.tables[cluster].store(transition)

		self.t += 1

	def sample(self, batch_size):

		s_batch = th.tensor([])
		a_batch = th.tensor([])
		r_batch = th.tensor([])
		sp_batch = th.tensor([])
		d_batch = th.tensor([])

		for k in range(self.k_events):
			num_samples = int(min(batch_size/self.k_events, len(self.tables[k].table)))
			
			s, a, r, s_p, d = self.tables[k].sample(num_samples)
			s_batch = th.cat([s_batch, s])
			a_batch = th.cat([a_batch, a])
			r_batch = th.cat([r_batch, r])
			sp_batch = th.cat([sp_batch, s_p])
			d_batch = th.cat([d_batch, d])
			
		return s_batch, a_batch, r_batch, sp_batch, d_batch

import torch as th
from collections import deque
from sklearn.cluster import KMeans
from cardio_rl.transitions import TorchTransition
from cardio_rl.buffers.er_buffer import ErTable


class IeTable:
    def __init__(
        self,
        network,
        capacity=1_000_000,
        transition_func=TorchTransition,
        k_events=4,
        cluster_interval=10,
        percent=0.5,
    ):
        self.tables = [
            ErTable(capacity // k_events, transition_func) for _ in range(k_events)
        ]
        self.buffer = ErTable(capacity, transition_func)
        self.k_events = k_events
        self.all_encoded_transitions = deque(maxlen=capacity)  # []
        self.encoder = lambda x: x
        self.cluster_interval = cluster_interval
        self.percent = percent
        self.t = 0
        self.network = network

    def __len__(self):
        lengths = [len(self.tables[k]) for k in range(self.k_events)]
        return sum(lengths)

    def store(self, transition):
        self.buffer.store(transition)

        if self.t % self.cluster_interval == 0:
            self.kmeans = KMeans(n_clusters=self.k_events, n_init="auto").fit(
                self.all_encoded_transitions
            )

        state, action, *_ = transition

        """With raw, concatonated vectors"""
        # z = np.concatenate([state, action], axis=0)
        """With state embedding"""
        z = self.network.embedding(th.from_numpy(state)).detach().numpy()

        self.all_encoded_transitions.append(z)
        cluster = self.kmeans.predict([z]).squeeze(0)

        self.tables[cluster].store(transition)

        self.t += 1

    def sample(self, batch_size):
        from_buffer = int(batch_size * (self.percent))
        from_tables = int((batch_size - from_buffer) // self.k_events)
        from_buffer += int(
            batch_size - (from_tables * self.k_events)
        )  # ensure batch_size samples are received

        s_batch, a_batch, r_batch, sp_batch, d_batch, _ = self.buffer.sample(
            from_buffer
        )

        # lengths = []

        for k in range(self.k_events):
            # num_samples = int(min(, len(self.tables[k].table)))

            # Don't sample empty tables
            if len(self.tables[k]) > 0:
                s, a, r, s_p, d, _ = self.tables[k].sample(int(from_tables))
                s_batch = th.cat([s_batch, s])
                a_batch = th.cat([a_batch, a])
                r_batch = th.cat([r_batch, r])
                sp_batch = th.cat([sp_batch, s_p])
                d_batch = th.cat([d_batch, d])

            # lengths.append(len(self.tables[k]))

        # print(lengths)

        return s_batch, a_batch, r_batch, sp_batch, d_batch, _

import gymnasium as gym
import numpy as np
from gymnasium import Env

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition

"""
Links:
* https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
* https://github.com/google/dopamine/blob/master/dopamine/replay_memory/prioritized_replay_buffer.py

Both implementations have maximum priority as the maximum ever seen, not current maximum

Should document implementation details as they were important!
"""


class SumTree:
    """_summary_"""

    def __init__(self, size):
        self.n = len(size)
        self.data = np.zeros(self.n)
        self.tree = np.zeros((2 * self.n) - 1)

    def _update(self, data_idx, value):
        # value = np.random.randint(0, 10)  # TODO: never forget I left this in for weeks...

        self.data[data_idx] = value

        idx = data_idx + self.n - 1  # child index in tree array
        change = value - self.tree[idx]

        self.tree[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def update(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self._update(i, p)

    def _sample(self, cumsum):
        idx = 0
        while 2 * idx + 1 < len(self.tree):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.tree[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.tree[left]

        data_idx = idx - self.n + 1
        return data_idx

    def sample(self, batch_size):
        span = self.total / batch_size
        idxs = [
            self._sample(np.random.uniform(i * span, (i + 1) * span))
            for i in range(batch_size)
        ]
        return np.array(idxs)

    @property
    def total(self):
        return self.tree[0]


class PrioritisedBuffer(TreeBuffer):
    """Prioritised Tree buffer has an additional key for storing priorities
    that are then used to calculate probabilities for categorical sampling of
    indices. Includes a couple of.

    important implementation details outlined in the paper, such as:     * Using a
    sumtree for efficient time complexity.     * Stratified sampling using uniform
    distribution between 0 and sum of probabilities.

    This specific implementation is most similar to the one in Dopamine, but... TODO:
    finish

    Internal keys: s, a, r, d, or one of the extra specs provided.

    Attributes:     pos: Moving record of the current position to store transitions.
    capacity: Maximum size of buffer.     full: Is the replay buffer full or not. table:
    The main dictionary containing transitions.
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        batch_size: int = 32,
        n_steps: int = 1,
        alpha: float = 0.5,
        beta: float = 1.0,  # Fixed schedule from dopamine
        eps: float = 1e-2,  # No mention in paper, going off of implementations TODO: check dopamine
    ):
        self.sumtree = SumTree(np.zeros(capacity))
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.max_p = 1.0
        super().__init__(env, capacity, extra_specs, batch_size, n_steps)

    def store(self, data: Transition, num: int):
        p = np.full(num, self.max_p**0.5)
        idxs = super().store(data, num)
        self.sumtree.update(idxs, p)
        return idxs

    def _sample(
        self,
        batch_size: int | None = None,
        sample_indxs: np.ndarray | None = None,
    ):
        if sample_indxs is None:
            sample_indxs = self.sumtree.sample(batch_size)

        data = super()._sample(sample_indxs=sample_indxs)

        probs = self.sumtree.data[sample_indxs] / self.sumtree.total
        probs = np.expand_dims(probs, -1)
        data.update({"p": probs})
        w = (1 / (len(self) * probs)) ** self.beta
        w = w / np.max(w)
        data.update({"w": w})
        return data

    def update(self, data: dict):
        p = np.sqrt(data.pop("p")) + self.eps
        if len(p.shape) == 2:
            p = np.squeeze(p, -1)
        self.max_p = max(p.max(), self.max_p)
        self.sumtree.update(data["idxs"], p)
        super().update(data)


def main():
    env = gym.make("CartPole-v1")
    buffer = PrioritisedBuffer(env, capacity=10, batch_size=2)

    print(buffer.sumtree.data)

    s, _ = env.reset()

    for i in range(4):
        a = env.action_space.sample()
        s_p, r, d, t, _ = env.step(a)
        d = d or t

        transition = {
            "s": np.array([s]),
            "a": np.array([a]),
            "r": np.array([r]),
            "s_p": np.array([s_p]),
            "d": np.array([int(d)]),
        }

        buffer.store(transition, 1)

        if d:
            s_p, _ = env.reset()

        s = s_p

    print(buffer.sumtree.data)
    print(buffer.sample()["idxs"])


if __name__ == "__main__":
    main()

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import Env

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition

"""
Sum tree is extremely slow, 1 iteration a second...

Current understanding is very wrong, there are lots mroe technical details
in the paper, need to review further.

Link: https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
"""


class SumTree:
    def __init__(self, size):
        self.n = len(size)
        self.data = np.zeros(self.n)
        self.tree = np.zeros((2 * self.n) - 1)

    def update(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self._update(i, p)

    def _update(self, data_idx, value):
        value = np.random.randint(0, 10)

        self.data[data_idx] = value

        idx = data_idx + self.n - 1  # child index in tree array
        change = value - self.tree[idx]

        self.tree[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def sample(self, batch_size):
        seg_length = self.total / batch_size
        idxs = []
        for i in range(batch_size):
            uni = np.random.uniform(i * seg_length, (i + 1) * seg_length)
            idxs.append(self._sample(uni))

        return idxs

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

    @property
    def total(self):
        return self.tree[0]

    @property
    def max(self):
        return self._sample(self.total)


class PrioritisedBuffer(TreeBuffer):
    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        n_steps: int = 1,
        alpha: float = 0.5,  # Rainbow default
        beta: float = 1.0,  # Fixed schedule from dopamine
    ):
        self.sumtree = SumTree(np.zeros(capacity))
        self.alpha = alpha
        self.beta = beta
        super().__init__(env, capacity, extra_specs, n_steps)

    def store(self, data: Transition, num: int):
        if self.__len__() == 0:
            max_p = 1.0
        else:
            max_p = np.power(self.sumtree.max, 1 / self.alpha)  # Unsure about this
            # max_p = self.sumtree.max

        p = np.ones(num) * max_p
        idxs = super().store(data, num)
        self.sumtree.update(idxs, p)
        return idxs

    def sample(
        self,
        batch_size: Optional[int] = None,
        sample_indxs: Optional[np.ndarray] = None,
    ):
        sample_indxs = self.sumtree.sample(batch_size)
        data = super().sample(sample_indxs=sample_indxs)

        priorities = self.sumtree.data[sample_indxs]
        probs = np.expand_dims(priorities / self.sumtree.total, -1)
        data.update({"p": probs})

        w = (self.__len__() * probs) ** -self.beta
        w = w / np.max(w)
        data.update({"w": w})
        return data

    def update(self, data: dict):
        p = np.squeeze(data.pop("p"), -1)
        p = np.power(p, self.alpha)
        self.sumtree.update(data["idxs"], p)
        super().update(data)


def main():
    env = gym.make("CartPole-v1")
    buffer = PrioritisedBuffer(env, capacity=10)

    print(buffer("p"))

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

    print(buffer("p"))
    print(buffer.sample(2)["idxs"])


if __name__ == "__main__":
    main()

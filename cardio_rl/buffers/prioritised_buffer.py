import functools

import gymnasium as gym
import jax
import numpy as np
from gymnasium import Env

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition


class PrioritisedBuffer(TreeBuffer):
    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        n_steps: int = 1,
    ):
        super().__init__(env, capacity, extra_specs, n_steps)
        priority_shape = self.base_shape[0]
        self.table.update({"p": np.zeros((priority_shape, 1))})

    def store(self, transition: Transition, num: int):
        def _place(arr, x, idx):
            if len(x.shape) == 1:
                x = np.expand_dims(x, -1)

            arr[idx] = x
            return arr

        max_prob = 1.0  # max(self.table["p"])
        transition.update({"p": np.full(shape=(num, 1), fill_value=max_prob)})
        idxs = np.arange(self.pos, self.pos + num) % self.capacity
        place = functools.partial(_place, idx=idxs)

        self.table = jax.tree.map(place, self.table, transition)

        self.pos += num
        if self.pos >= self.capacity:
            self.full = True
            self.pos = self.pos % self.capacity

    def sample(self, batch_size: int):
        sample_size = int(min(batch_size, self.__len__()))
        priorities = np.squeeze(self.table["p"][np.arange(self.__len__())], -1)
        probs = priorities / np.sum(
            priorities
        )  # Need to implement the actual probability calculation
        sample_indxs = np.random.choice(self.__len__(), sample_size, p=probs)
        batch: dict = jax.tree.map(lambda arr: arr[sample_indxs], self.table)
        batch.update({"idxs": sample_indxs})
        return batch


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

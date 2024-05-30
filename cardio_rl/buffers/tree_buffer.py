import functools
import jax
import random
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from gymnasium import Env


class TreeBuffer:
    def __init__(self, env: Env, capacity: int = 1_000_000, extra_specs: dict = {}):
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

        self.table = {
            "s": np.zeros((self.capacity, *obs_dims), dtype=obs_space.dtype),
            "a": np.zeros(
                (
                    self.capacity,
                    act_dim,
                ),
                dtype=act_space.dtype,
            ),
            "r": np.zeros((self.capacity, 1)),
            "s_p": np.zeros((self.capacity, *obs_dims), dtype=obs_space.dtype),
            "d": np.zeros((self.capacity, 1)),
        }

        if extra_specs:
            extras = {}
            for key, value in extra_specs.items():
                shape = [capacity] + value
                extras.update({key: np.zeros(shape)})

            self.table.update(extras)

    def __len__(self):
        if self.full:
            return self.capacity
        return self.pos

    def store(self, transition: dict, num: int):
        def _place(arr, x, idx):
            """
            Instead of reshaping in this function maybe look into doing it in the gatherer
            """
            if len(x.shape) == 1:
                x = np.expand_dims(x, -1)

            arr[idx] = x
            return arr

        """
		Need to verify this works as expected and there's no silent bugs
		"""

        idxs = np.arange(self.pos, self.pos + num) % self.capacity
        place = functools.partial(_place, idx=idxs)
        self.table = jax.tree.map(place, self.table, transition)

        self.pos += num
        if self.pos >= self.capacity:
            self.full = True
            self.pos = self.pos % self.capacity

    def sample(self, batch_size: int):
        sample_size = int(min(batch_size, self.__len__()))
        sample_indxs = random.sample(range(self.__len__()), sample_size)
        batch: dict = jax.tree.map(lambda arr: arr[sample_indxs], self.table)
        batch.update({"idxs": sample_indxs})
        return batch

    def update(self, new_dict: dict):
        """
        Should be overhauled to be able to see and edit the entire table, or based on indices etc.
        """
        self.table.update(new_dict)


def main():
    env = gym.make("CartPole-v1")
    buffer = TreeBuffer(env, capacity=100)

    s, _ = env.reset()

    for i in range(3_000):
        a = env.action_space.sample()
        s_p, r, d, t, _ = env.step(a)
        d = d or t

        transition = {
            "s": s,
            "a": a,
            "r": r,
            "s_p": s_p,
            "d": d,
        }

        buffer.store(transition)

        if d:
            s_p, _ = env.reset()

        s = s_p

    print(buffer.sample(4))


if __name__ == "__main__":
    main()

import functools
import random

import gymnasium as gym
import jax
import numpy as np
from gymnasium import Env, spaces

from cardio_rl.types import Transition


class TreeBuffer:
    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        n_steps: int = 1,
    ):
        obs_space = env.observation_space
        obs_dims = obs_space.shape

        act_space = env.action_space

        if isinstance(act_space, spaces.Box):
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, spaces.Discrete):
            act_dim = 1

        self.pos = 0
        self.capacity = capacity
        self.base_shape = [capacity]

        # if traj_len > 1:
        #     base_shape += [traj_len]

        self.full = False

        self.table: dict = {
            "s": np.zeros((*self.base_shape, *obs_dims), dtype=obs_space.dtype),
            "a": np.zeros(
                (
                    *self.base_shape,
                    act_dim,
                ),
                dtype=act_space.dtype,
            ),
            "r": np.zeros((*self.base_shape, n_steps)),
            "s_p": np.zeros((*self.base_shape, *obs_dims), dtype=obs_space.dtype),
            "d": np.zeros((*self.base_shape, 1)),
        }

        if extra_specs:
            extras = {}
            for key, value in extra_specs.items():
                shape = [capacity] + value
                extras.update({key: np.zeros(shape)})

            self.table.update(extras)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def __call__(self, key: str):
        return self.table[key]

    def store(self, batch: Transition, num: int):
        def _place(arr, x, idx):
            if len(x.shape) == 1:
                x = np.expand_dims(x, -1)

            arr[idx] = x
            return arr

        idxs = np.arange(self.pos, self.pos + num) % self.capacity
        place = functools.partial(_place, idx=idxs)
        self.table = jax.tree.map(place, self.table, batch)

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

    def update(self, data: dict):
        idxs = data.pop("idxs")
        for key, val in data.items():
            if key in self.table:
                self.table[key][idxs] = val


def main():
    env = gym.make("CartPole-v1")
    buffer = TreeBuffer(env, capacity=10)

    print(buffer("p"))
    buffer.update(
        {
            "idxs": np.arange(8),
            "p": np.expand_dims(np.sin(np.arange(8)), -1),
        }
    )
    print(buffer("p"))
    exit()

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

import functools
from typing import Optional

import jax
import numpy as np
from gymnasium import Env, spaces

from cardio_rl.types import Transition


class TreeBuffer:
    """Extensible replay buffer that stores transitions as a dictionary.

    Longer class information...

    s, a, r, s_p, d, or one of the extra specs provided.

    Attributes:
        pos: Moving record of the current position to store transitions.
        capacity: Maximum size of buffer.
        full: Is the replay buffer full or not.
        table: The main dictionary containing transitions.
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        n_steps: int = 1,
    ):
        """_summary_

        Args:
            env (Env): Gymnasium environment used to construct the buffer shapes.
            capacity (int, optional): Maximum size of buffer. Defaults to 1_000_000.
            extra_specs (dict, optional): Any extra elements to store. Defaults to {}.
            n_steps (int, optional): Environment steps per transition. Defaults to 1.
        """

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

        self.table: dict = {
            "s": np.zeros((capacity, *obs_dims), dtype=obs_space.dtype),  # type: ignore
            "a": np.zeros(
                (
                    capacity,
                    act_dim,
                ),
                dtype=act_space.dtype,
            ),
            "r": np.zeros((capacity, n_steps)),
            "s_p": np.zeros((capacity, *obs_dims), dtype=obs_space.dtype),  # type: ignore
            "d": np.zeros((capacity, 1)),
        }

        if extra_specs:
            extras = {}
            for key, value in extra_specs.items():
                shape = [capacity] + value
                extras.update({key: np.zeros(shape)})

            self.table.update(extras)

    def __len__(self) -> int:
        """The current amount of transitions stored in the internal
        table.

        Returns:
            An integer describing the current length of stored data.
        """
        return self.capacity if self.full else self.pos

    def __call__(self, key: str) -> np.array:
        """Access specific MDP elements in the internal table.

        Args:
            key (str): The key of the element to be accessed.

        Returns:
            The entire numpy array of the requested element from the
            internal table. Will contain zeros in indices not yet
            used for storage.
        """
        return self.table[key]

    def store(self, data: Transition, num: int) -> np.ndarray:
        """Store the given transitions in the replay buffer. The buffer
        is circular and determines the indices to be used before placing
        the MDP elements in the internal table. Also accounts for storing
        any extra specifications.

        Args:
            data (Transition): A dictionary containing 1 or more
                transitions worth of MDP elements.
            num (int): The amount of transitions contained in the
                data.

        Returns:
            The entire numpy array of the indices used to store
            the provided data.
        """

        def _place(arr, x, idx):
            if len(x.shape) == 1:
                x = np.expand_dims(x, -1)

            arr[idx] = x
            return arr

        idxs = np.arange(self.pos, self.pos + num) % self.capacity
        place = functools.partial(_place, idx=idxs)
        self.table = jax.tree.map(place, self.table, data)

        self.pos += num
        if self.pos >= self.capacity:
            self.full = True
            self.pos = self.pos % self.capacity

        return idxs

    def sample(
        self,
        batch_size: Optional[int] = None,
        sample_indxs: Optional[np.ndarray] = None,
    ) -> Transition:
        """Sample batch_size number of indices between 0 and the current
        length of the replay buffer. Take each corresponding transition
        and compile into a new dictionary.

        Args:
            batch_size (int): The number of samples to take from the internal table.

        Returns:
            A dictionary containing the MDP elements of transitions
            sampled from the buffer as well as the indices used (accessed
            using the "idxs" key).
        """
        # TODO: raise an error/warning when batch_size and sample_indxs are both passed.
        if batch_size:
            sample_indxs = np.random.randint(
                low=0, high=self.__len__(), size=batch_size
            )

        batch: dict = jax.tree.map(lambda arr: arr[sample_indxs], self.table)
        batch.update({"idxs": sample_indxs})
        return batch

    def update(self, data: dict):
        """Update specific keys and indices in the internal table with
        new or updated data, e.g. latest priorities.

        Args:
            data (dict): A dictionary containing an "idxs" key with
                indices to update and keys with the updated values.
        """
        idxs = data.pop("idxs")
        for key, val in data.items():
            if key in self.table:
                self.table[key][idxs] = val


# def main():
#     env = gym.make("CartPole-v1")
#     buffer = TreeBuffer(env, capacity=10)

#     print(buffer("p"))
#     buffer.update(
#         {
#             "idxs": np.arange(8),
#             "p": np.expand_dims(np.sin(np.arange(8)), -1),
#         }
#     )
#     print(buffer("p"))
#     exit()

#     s, _ = env.reset()

#     for i in range(3_000):
#         a = env.action_space.sample()
#         s_p, r, d, t, _ = env.step(a)
#         d = d or t

#         transition = {
#             "s": s,
#             "a": a,
#             "r": r,
#             "s_p": s_p,
#             "d": d,
#         }

#         buffer.store(transition)

#         if d:
#             s_p, _ = env.reset()

#         s = s_p

#     print(buffer.sample(4))


# if __name__ == "__main__":
#     main()

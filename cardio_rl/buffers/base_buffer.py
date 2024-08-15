from typing import Optional

import numpy as np
from gymnasium import Env, spaces

from cardio_rl.types import Transition


class BaseBuffer:
    """Simple replay buffer that stores transitions as a dictionary.

    Longer class information...

    s, a, r, s_p, and d.

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

        self.s = np.zeros((capacity, *obs_dims), dtype=obs_space.dtype)  # type: ignore
        self.a = np.zeros(
            (
                capacity,
                act_dim,
            ),
            dtype=act_space.dtype,
        )
        self.r = np.zeros((capacity, n_steps))
        self.s_p = np.zeros((capacity, *obs_dims), dtype=obs_space.dtype)  # type: ignore
        self.d = np.zeros((capacity, 1))

    def __len__(self) -> int:
        """The current amount of transitions stored in the internal
        table.

        Returns:
            An integer describing the current length of stored data.
        """
        return self.len

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

        idxs = np.arange(self.pos, self.pos + num) % self.capacity

        self.s[idxs] = data["s"]
        self.a[idxs] = np.expand_dims(data["a"], -1)
        if data["r"].shape == 1:
            r = np.expand_dims(data["r"], -1)
        else:
            r = data["r"]

        self.r[idxs] = r
        self.s_p[idxs] = data["s_p"]
        self.d[idxs] = np.expand_dims(data["d"], -1)

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
        if batch_size and sample_indxs is None:
            sample_indxs = np.random.randint(low=0, high=self.len, size=batch_size)

        batch: dict = {
            "s": self.s[sample_indxs],
            "a": self.a[sample_indxs],
            "r": self.r[sample_indxs],
            "s_p": self.s_p[sample_indxs],
            "d": self.d[sample_indxs],
        }
        batch.update({"idxs": sample_indxs})
        return batch

    def update(self, data: dict):
        """Update specific keys and indices in the internal table with
        new or updated data, e.g. latest priorities.

        Args:
            data (dict): A dictionary containing an "idxs" key with
                indices to update and keys with the updated values.
        """
        del data
        pass

    @property
    def len(self):
        return self.capacity if self.full else self.pos

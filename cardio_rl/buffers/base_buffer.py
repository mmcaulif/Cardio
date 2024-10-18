from typing import Optional

import numpy as np
from gymnasium import Env, spaces

from cardio_rl.types import Transition


class BaseBuffer:
    """Simple replay buffer that stores transitions as numpy arrays in
    individual attributes.

    Internal keys: s, a, r, s_p, or d.

    Attributes:
        pos: Moving value of the current position to store transitions.
        capacity: Maximum size of buffer.
        full: Is the replay buffer full or not.
        s: Array of shape [capacity, |s|] containing the transitions state.
        a: Array of shape [capacity, 1] (or [capacity, |a|] for
            continuous environments) containing the transitions action.
        r: Array of shape [capacity, n_steps] containing the reward
            received.
        s_p: Array of shape [capacity, |s|] containing the transitions
            next state.
        d: Array of shape [capacity, 1] containing the terminal boolean.
        len: Number of transitions stored in the buffer.
    """

    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        n_steps: int = 1,
        trajectory: int = 1,
    ):
        """Initialises the replay buffer, automatically building the table
        using provided parameters and an environment.

        Args:
            env (Env): Gymnasium environment used to construct the buffer shapes.
            capacity (int, optional): Maximum size of buffer. Defaults to 1_000_000.
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
        self.n_steps = n_steps
        self.trajectory = trajectory

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
        """The current amount of transitions stored in the internal table.

        Returns:
            An integer describing the current length of stored data.
        """
        return self.len

    def store(self, data: Transition, num: int) -> np.ndarray:
        """Store the given transitions in the replay buffer. The buffer is
        circular and determines the indices to be used before placing the MDP
        elements in the internal table.

        Args:
            data (Transition): A dictionary containing 1 or more
                transitions worth of MDP elements.
            num (int): The amount of transitions contained in the
                data.

        Returns:
            np.ndarray: The entire numpy array of the indices used to store
                the provided data.
        """

        idxs = np.arange(self.pos, self.pos + num) % self.capacity

        self.s[idxs] = data["s"]
        self.a[idxs] = data["a"]
        if data["r"].shape == 1:
            r = np.expand_dims(data["r"], -1)
        else:
            r = data["r"]

        self.r[idxs] = r
        self.s_p[idxs] = data["s_p"]
        self.d[idxs] = data["d"]

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
        """Sample batch_size number of indices between 0 and the current length
        of the replay buffer, or use provided indices. Take each corresponding
        transition and compile into a new dictionary.

        Args:
            batch_size (int): The number of samples to take from the internal table.

        Raises:
            ValueError: Trying to pass both a batch_size and sample_indxs, can only use one.

        Returns:
            Transition: A dictionary containing the MDP elements of transitions
                sampled from the buffer as well as the indices used (accessed
                using the "idxs" key).
        """

        if batch_size and sample_indxs:
            raise ValueError(
                "Passing both a batch size and indices to sample method, please only provide one"
            )

        if batch_size and sample_indxs is None:
            sample_indxs = np.random.randint(
                low=0, high=self.len - (self.trajectory - 1), size=batch_size
            )

        assert sample_indxs

        batch: dict = {}

        if self.trajectory != 1:
            batch.update(
                {
                    "s": np.stack(
                        [self.s[idx : idx + self.trajectory] for idx in sample_indxs]
                    ),
                    "a": np.stack(
                        [self.a[idx : idx + self.trajectory] for idx in sample_indxs]
                    ),
                    "r": np.stack(
                        [self.d[idx : idx + self.trajectory] for idx in sample_indxs]
                    ),
                    "s_p": np.stack(
                        [self.s_p[idx : idx + self.trajectory] for idx in sample_indxs]
                    ),
                    "d": np.stack(
                        [self.d[idx : idx + self.trajectory] for idx in sample_indxs]
                    ),
                }
            )
        else:
            batch.update(
                {
                    "s": self.s[sample_indxs],
                    "a": self.a[sample_indxs],
                    "r": self.r[sample_indxs],
                    "s_p": self.s_p[sample_indxs],
                    "d": self.d[sample_indxs],
                }
            )
        batch.update({"idxs": sample_indxs})
        return batch

    def update(self, data: dict):
        """Dummy function in BaseBuffer but implemented in children classes.
        Used to Update specific keys and indices in the internal table with new
        or updated data, e.g. latest priorities.

        Raise:
            UserWarning: Passing data to this dummy update function.

        Args:
            data (dict): A dictionary containing an "idxs" key with
                indices to update and keys with the updated values.
        """
        del data
        raise UserWarning(
            "Passing update data to the base buffer, use tree buffer instead"
        )

    @property
    def len(self):
        """The current amount of transitions stored in the internal table.

        Returns:
            An integer describing the current length of stored data.
        """
        return self.capacity if self.full else self.pos

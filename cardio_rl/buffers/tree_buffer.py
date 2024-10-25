import functools

import jax
import numpy as np
from gymnasium import spaces

from cardio_rl.buffers.base_buffer import BaseBuffer
from cardio_rl.types import Environment, Transition


class TreeBuffer(BaseBuffer):
    """Extensible replay buffer that stores transitions as a dictionary and
    allows for extra elements to be stored and sampled per transition.

    Internal keys: s, a, r, s_p, d, or one of the extra specs provided.

    Attributes:
        pos: Moving record of the current position to store transitions.
        capacity: Maximum size of buffer.
        full: Is the replay buffer full or not.
        table: The main dictionary containing transitions.
    """

    def __init__(
        self,
        env: Environment,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        batch_size: int = 32,
        n_steps: int = 1,
        trajectory: int = 1,
        n_batches: int = 1,
    ):
        """Initialises the replay buffer, automatically building the table
        using provided parameters, an environment and optional extras.

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
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.trajectory = trajectory
        self.n_batches = n_batches

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
            "r": np.zeros((capacity, n_steps), dtype=np.float32),
            "s_p": np.zeros((capacity, *obs_dims), dtype=obs_space.dtype),  # type: ignore
            "d": np.zeros((capacity, 1), dtype=bool),
        }

        if extra_specs:
            extras = {}
            for key, value in extra_specs.items():
                shape = [capacity] + value
                extras.update({key: np.zeros(shape)})

            self.table.update(extras)

    def __call__(self, key: str) -> np.ndarray:
        """Access specific MDP table in the internal table using the
        corresponding key.

        Args:
            key (str): The key of the element to be accessed.

        Returns:
            np.ndarray: The entire numpy array of the requested element from the
                internal table. Will contain zeros in indices not yet
                used for storage.
        """
        return self.table[key]

    def store(self, data: Transition, num: int) -> np.ndarray:
        """Store the given transitions in the replay buffer. The buffer is
        circular and determines the indices to be used before placing the MDP
        elements in the internal table. Also accounts for storing any extra
        specifications.

        Args:
            data (Transition): A dictionary containing 1 or more
                transitions worth of MDP elements.
            num (int): The amount of transitions contained in the
                data.

        Returns:
            np.ndarray: The entire numpy array of the indices used to store
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

    def _sample(
        self,
        batch_size: int | None = None,
        sample_indxs: np.ndarray | None = None,
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

        if batch_size:
            sample_indxs = np.random.randint(
                low=0, high=len(self) - (self.trajectory - 1), size=batch_size
            )

        def get_trajectories(arr):
            if self.trajectory != 1:
                trajectory_samples = np.stack(
                    [arr[idx : idx + self.trajectory] for idx in sample_indxs]
                )
            else:
                trajectory_samples = arr[sample_indxs]

            return trajectory_samples

        batch: dict = jax.tree.map(lambda arr: get_trajectories(arr), self.table)
        batch.update({"idxs": sample_indxs})
        return batch

    def update(self, data: dict):
        """Update specific keys and indices in the internal table with new or
        updated data, e.g. latest priorities.

        Args:
            data (dict): A dictionary containing an "idxs" key with
                indices to update and keys with the updated values.
        """
        idxs = data.pop("idxs")
        for key, val in data.items():
            if key in self.table:
                self.table[key][idxs] = val

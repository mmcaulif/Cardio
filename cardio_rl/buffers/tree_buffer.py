"""TreeBuffer class, main experience replay buffer used in Cardio."""

import functools

import jax
import numpy as np
from gymnasium import spaces

from cardio_rl.buffers.base_buffer import BaseBuffer
from cardio_rl.types import Environment, Transition


class TreeBuffer(BaseBuffer):
    """Buffer that stores transitions in a pytree."""

    def __init__(
        self,
        env: Environment,
        capacity: int = 1_000_000,
        extra_specs: dict | None = None,
        batch_size: int = 32,
        n_steps: int = 1,
        trajectory: int = 1,
        n_batches: int = 1,
    ):
        """Initialise the tree buffer with any additional entries.

        Stores data a dictionary where keys correspond to numpy arrays
        with the stored data. Users can define extra specifications to
        store entries beyodn the traditional state, action reward etc.
        To do so, provide a dictionary with the keys corresponding to
        the key that the agent will give the extras in, and the value
        being the shape of a single entry (cardio will create the full
        column). For example: extra_specs = {"log_probs": [1]} will
        will create a key value pair in the internal table dictionary
        where "log_probs" corresponds to a numpy array with shape
        [capacity, 1]. The tree buffer uses jax.tree.map to easily
        perform operations like storing or sampling from the buffer.

        Args:
            env (Env): A gymnasium environment, used to determine
                shapes.
            capacity (int, optional): Maximum number of transitions
                for the replay buffer, will pop old data once capacity
                has been exceeded. Defaults to 1_000_000.
            extra_specs (dict | None, optional): Additional entries to add
                to the replay buffer. Values must correspond to the
                shape. Defaults to None.
            batch_size (int, optional): Batch size to sample from the
                buffer. If set to None, the sample method will expect
                sample indices to be provided. Defaults to 32.
            n_steps (int, optional): Number of environment steps that a
                transition represents, sampled transitions take the
                form: {s_t, a_t, r_t+r_(t+1)+...+r_(t+n), s_(t+n),
                d_(t+n)}. Defaults to 1.
            trajectory (int, optional): How many sequential transitions
                to take per sample index. Defaults to 1.
            n_batches (int, optional): How many batches of batch_size
                samples to do at sample time, requires a batch_size
                be provided. Defaults to 1.
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

        if extra_specs is not None:
            extras = {}
            for key, value in extra_specs.items():
                shape = [capacity] + value
                extras.update({key: np.zeros(shape)})

            self.table.update(extras)

    def store(self, data: Transition, num: int) -> np.ndarray:
        """Store the given transitions in the replay buffer.

        The buffer is circular and determines the indices to be used
        before placing the MDP elements in the internal table.

        Args:
            data (Transition): A dictionary containing 1 or more
                transitions worth of MDP elements.
            num (int): The amount of transitions contained in the data.

        Returns:
            np.ndarray: The entire numpy array of the indices used to
                store the provided data.
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
        """Randomly sample directly from the buffer.

        Private method for sampling from the buffer. Using a given
        batch_size number of random indices, or a provided array of
        indices, take each corresponding transition and compile into
        a new dictionary.

        Args:
            batch_size (int | None, optional): The number of samples
                to take from the internal table. Defaults to None.
            sample_indxs (np.ndarray | None, optional): A numpy array
                of indices to take from the buffer. Defaults to None.

        Raises:
            ValueError: Trying to pass both a batch_size and
                sample_indxs, can only use one.

        Returns:
            Transition: A dictionary containing the MDP elements of
                transitions sampled from the buffer as well as the
                indices used (accessed using the "idxs" key).
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
        """Update the data stored in the buffer.

        update specific keys and indices in the internal table with new
        or updated data, e.g. latest priorities. Only called if the
        agent's update step returns a non-empty dictionary. Must
        provide indices via a "idxs" key.

        Args:
            data (dict): A dictionary containing an "idxs" key with
                indices to update and keys with the updated values.

        Raises:
            ValueError: Provided the update method with data but no idxs key.
        """
        if "idxs" in data:
            idxs = data.pop("idxs")
            for key, val in data.items():
                if key in self.table:
                    self.table[key][idxs] = val
        else:
            raise ValueError(
                "Passing data to update the buffer but not supplying indices via an idxs key"
            )

    def get(self, key: str, include_empty: bool = False) -> np.ndarray:
        """Get a specific column from the internal table.

        Given a key, return the filled (or entire) column corresponding
        to the provided key.

        Args:
            key (str): Key in the internal table: s, a, r, s_p, d, or
                one provided in the extra specs.
            include_empty (bool, optional): Whether to return the whole
                column, or just the elements that have been filled so
                far. Defaults to False.

        Returns:
            np.ndarray: The column corresponding to the key argument.
        """
        if include_empty:
            return self.table[key]

        return self.table[key][: self.pos]

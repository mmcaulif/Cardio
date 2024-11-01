"""BaseBuffer class, parent class of other buffers in Cardio."""

import warnings

import numpy as np
from gymnasium import Env, spaces

from cardio_rl.types import Transition


class BaseBuffer:
    """Base replay buffer class that implements a simple buffer."""

    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        batch_size: int = 32,
        n_steps: int = 1,
        trajectory: int = 1,
        n_batches: int = 1,
    ):
        """Initialise the BaseBuffer.

        Stores data using numpy arrays in the s, a, r, s_p and d
        attributes. Acts as a simple example of an experience replay
        buffer.

        Args:
            env (Env): A gymnasium environment, used to determine
                shapes.
            capacity (int, optional): Maximum number of transitions
                for the replay buffer, will pop old data once capacity
                has been exceeded. Defaults to 1_000_000.
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
        """Length of the buffer.

        The current amount of transitions stored in the internal table.

        Returns:
            int: An integer describing the current length of stored data.
        """
        length = self.capacity if self.full else self.pos
        return length

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

        if batch_size and sample_indxs is None:
            sample_indxs = np.random.randint(
                low=0, high=len(self) - (self.trajectory - 1), size=batch_size
            )

        assert sample_indxs is not None

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

    def sample(
        self, sample_indxs: np.ndarray | None = None
    ) -> Transition | list[Transition]:
        """Randomly sample directly from the buffer.

        Method for sampling from the buffer. If no sample indices are
        provided, will use the internal batch_size attribute to
        determine how much random samples to take. If self.n_batches
        is greater than 1, it will sample the buffer for random
        indices that many times and provide a list of dictionaries.
        The indices used are stored in the "indxs" key in the resulting
        Transition dictionary.

        Args:
            sample_indxs (np.ndarray | None, optional): A numpy array
                of indices to take from the buffer. Defaults to None.

        Raises:
            ValueError: Provided sample indices but self.n_batches > 1.

        Returns:
            Transition: A dictionary or a list of dictionaries
                containing the MDP elements of transitions sampled
                from the buffer as well as the indices used (accessed
                using the "idxs" key).
        """
        if sample_indxs is None:
            k = min(self.batch_size, len(self))
            if self.n_batches > 1:
                return [self._sample(batch_size=k) for _ in range(self.n_batches)]
            else:
                return self._sample(batch_size=k)
        else:
            if self.n_batches > 1:
                raise ValueError("Passing sample indices when n_batches > 1")
            return self._sample(sample_indxs=sample_indxs)

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
        warnings.warn("Passing update data to the base buffer, use tree buffer instead")
        if "idxs" in data:
            del data
        else:
            raise ValueError(
                "Passing data to update the buffer but not supplying indices via an idxs key"
            )

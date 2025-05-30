"""TreeBuffer class, main experience replay buffer used in Cardio."""

import numpy as np

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Environment, Transition


class EffBuffer(TreeBuffer):
    """Buffer that stores transitions in a pytree."""

    def __init__(
        self,
        env: Environment,
        capacity: int = 1_000_000,
        extra_specs: dict | None = None,
        batch_size: int = 32,
        n_steps: int = 1,
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
            n_batches (int, optional): How many batches of batch_size
                samples to do at sample time, requires a batch_size
                be provided. Defaults to 1.
        """
        super().__init__(
            env=env,
            capacity=capacity,
            extra_specs=extra_specs,
            batch_size=batch_size,
            n_steps=n_steps,
            trajectory=1,
            n_batches=n_batches,
        )
        self.table.pop("s_p")

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
        data.pop("s_p")  # Remove the next state, not used in EffBuffer
        idxs = super().store(data, num)
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
            print(len(self))
            sample_indxs = np.random.randint(
                low=0, high=len(self) - (1 + (self.trajectory - 1)), size=batch_size
            )

        assert sample_indxs is not None, "No sample indices provided for sampling."
        batch = super()._sample(sample_indxs=sample_indxs)
        s_p_idxs = (sample_indxs + 1) % self.capacity
        batch.update({"s_p": self.table["s"][s_p_idxs]})
        return batch

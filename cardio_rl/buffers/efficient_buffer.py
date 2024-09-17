from typing import Optional

import numpy as np
from gymnasium import Env

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition


class EffBuffer(TreeBuffer):
    """Custom version of the tree buffer that only uses one state key
    internally for more memory efficiency. API is identical as the tree buffer
    but unfortunately is not compatible with n-step returns (might be possible
    in the future). Implementation is to be sanity checked.

    In a deterministic setting this should perform the same as tree buffer
    as the sampling indices is the same, will need to investigate...

    Internal keys: s, a, r, d, or one of the extra specs provided.

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
        super().__init__(env, capacity, extra_specs, n_steps)
        self.table.pop("s_p")

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

        s_p = data.pop("s_p")
        idxs = super().store(data, num)
        idxs = (idxs + 1) % self.capacity
        self.table["s"][idxs] = s_p
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

        batch = super().sample(batch_size, sample_indxs)
        idxs = batch["idxs"]
        idxs = (idxs + 1) % self.capacity
        s_p = self.table["s"][idxs]
        batch.update({"s_p": s_p})
        return batch

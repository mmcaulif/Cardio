"""TreeBuffer class, main experience replay buffer used in Cardio."""

import jax
import numpy as np

from cardio_rl.buffers.tree_buffer import TreeBuffer


class MixedBuffer(TreeBuffer):
    """Buffer that stores transitions in a pytree."""

    def _sample(
        self,
        batch_size: int | None = None,
        sample_indxs: np.ndarray | None = None,
    ):
        recent_idx = (self.pos - 1) % self.capacity

        if batch_size and sample_indxs:
            raise ValueError(
                "Passing both a batch size and indices to sample method, please only provide one"
            )

        if batch_size:
            sample_indxs = np.random.randint(
                low=0, high=len(self) - (self.trajectory - 1), size=batch_size - 1
            )
            sample_indxs = np.concatenate(([recent_idx], sample_indxs))

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

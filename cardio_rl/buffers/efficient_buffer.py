from typing import Optional

import numpy as np
from gymnasium import Env

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition


class EffBuffer(TreeBuffer):
    """In a deterministic setting this should perform the same as tree buffer
    as the sampling indices is the same, will need to investigate...

    * Need to sanity check this works as intended
    * Currently incompatible with n_step collection, explore solutions
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
        batch = super().sample(batch_size, sample_indxs)
        idxs = batch["idxs"]
        idxs = (idxs + 1) % self.capacity
        s_p = self.table["s"][idxs]
        batch.update({"s_p": s_p})
        return batch


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

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import Env

from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.types import Transition


class PrioritisedBuffer(TreeBuffer):
    def __init__(
        self,
        env: Env,
        capacity: int = 1_000_000,
        extra_specs: dict = {},
        n_steps: int = 1,
    ):
        extra_specs.update({"p": [1]})
        super().__init__(env, capacity, extra_specs, n_steps)

    def store(self, transition: Transition, num: int):
        if self.__len__() == 0:
            max_p = 0.0
        else:
            # TODO: replace with sum tree
            priorities: np.ndarray = self.table["p"][: self.__len__()]
            max_p = priorities.max()

        p = np.ones(num) * max_p
        transition.update({"p": p})
        super().store(transition, num)

    def sample(
        self,
        batch_size: Optional[int] = None,
        sample_indxs: Optional[np.ndarray] = None,
    ):
        del sample_indxs

        # TODO: replace with sum tree
        # priorities: np.ndarray = self.table["p"][:self.__len__()]
        # max_p = priorities.max()

        # probs = priorities/(sum(priorities) + 1e-8)

        # sample_indxs = np.random.choice(self.__len__(), batch_size, p=probs)

        # return super().sample(sample_indxs=sample_indxs) # type: ignore

        return super().sample(batch_size)  # type: ignore


def main():
    env = gym.make("CartPole-v1")
    buffer = PrioritisedBuffer(env, capacity=10)

    print(buffer("p"))

    s, _ = env.reset()

    for i in range(4):
        a = env.action_space.sample()
        s_p, r, d, t, _ = env.step(a)
        d = d or t

        transition = {
            "s": np.array([s]),
            "a": np.array([a]),
            "r": np.array([r]),
            "s_p": np.array([s_p]),
            "d": np.array([int(d)]),
        }

        buffer.store(transition, 1)

        if d:
            s_p, _ = env.reset()

        s = s_p

    print(buffer("p"))
    print(buffer.sample(2)["idxs"])


if __name__ == "__main__":
    main()

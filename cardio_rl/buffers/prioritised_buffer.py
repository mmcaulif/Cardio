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
        alpha: float = 0.5,  # Rainbow default
        beta: float = 1.0,  # Fixed schedule from dopamine
    ):
        extra_specs.update({"p": [1]})
        self.alpha = alpha
        self.beta = beta
        super().__init__(env, capacity, extra_specs, n_steps)

    def store(self, transition: Transition, num: int):
        if self.__len__() == 0:
            # Paper has p_1 = 1, other implementations have smaller
            # values but these seem to perform much worse.
            max_p = 1.0
        else:
            # TODO: replace with sum tree maximum
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
        priorities: np.ndarray = np.squeeze(self.table["p"][: self.__len__()], -1)

        priorities = np.power(
            priorities, self.alpha
        )  # TODO: double check correct value

        probs = priorities / sum(priorities)  # TODO: replace with sum tree total
        sample_indxs = np.random.choice(self.__len__(), size=batch_size, p=probs)  # type: ignore
        data = super().sample(sample_indxs=sample_indxs)

        # TODO: confirm what is the correct way to do this...
        # in paper is N the current capacity, total capacity or the batch size...
        sample_p = probs[sample_indxs]
        w = (self.__len__() * sample_p) ** -self.beta
        w = w / np.max(w)
        data.update({"w": w})
        return data


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

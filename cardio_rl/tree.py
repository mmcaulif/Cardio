import jax
import numpy as np
from cardio_rl import Transition


def stack(tree_list: list[Transition]) -> Transition:
    return jax.tree.map(lambda *arr: np.stack(arr), *tree_list)

import jax
import numpy as np
from cardio_rl import Transition


def stack(tree_list: list[Transition], axis=0) -> Transition:
    return jax.tree.map(lambda *arr: np.stack(arr, axis=axis), *tree_list)


def concatenate(tree_list: list[Transition], axis=0) -> Transition:
    return jax.tree.map(lambda *arr: np.concatenate(arr, axis=axis), *tree_list)
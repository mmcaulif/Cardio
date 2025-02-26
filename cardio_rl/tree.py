"""Tree utils used throughout Cardio."""

import jax
import numpy as np

from cardio_rl.types import Transition


def stack(tree_list: list[Transition], axis=0) -> Transition:
    """Tree map of numpy's stack.

    Stack leaf elements in a pytree along an existing axis, most often a list of dictionaries.

    Args:
        tree_list (list[Transition]): List of dictionaries whose leaf elements are to be stacked.
            Keys in each dictionary must match and shapes at each respective key must be the same.
        axis (int, optional): The axis in the result array along which the input arrays are
            stacked. Defaults to 0.

    Returns:
        Transition: A dictionary where each key corresponds to the respective stacked elements.
    """
    return jax.tree.map(lambda *arr: np.stack(arr, axis=axis), *tree_list)


def concatenate(tree_list: list[Transition], axis=0) -> Transition:
    """Tree map of numpy's concatenate.

    Concatenate leaf elements in a pytree along an existing axis, most often a list of dictionaries.

    Args:
        tree_list (list[Transition]): List of dictionaries whose leaf elements are to be concatenated.
             Keys in each dictionary must match and shapes at each respective key must be the same.
        axis (int, optional): The axis in the result array along which the input arrays are
            concatenated. Defaults to 0.

    Returns:
        Transition: A dictionary where each key corresponds to the respective concatenated elements.
    """
    return jax.tree.map(lambda *arr: np.concatenate(arr, axis=axis), *tree_list)

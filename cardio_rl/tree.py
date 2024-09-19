import jax
import numpy as np

from cardio_rl.types import Transition


def stack(tree_list: list[Transition], axis=0) -> Transition:
    """Stack leaf elements in a pytree (in cardio, most often a list of
    dictionaries) along a given axis, creating new dimension.

    Args:
        tree_list (list[Transition]): List of dictionaries whose leaf
            elements are to be stacked. Keys in each dictionary must
            match and shapes at each respective key must be the same.
        axis (int, optional): The axis in the result array along which
            the input arrays are stacked. Defaults to 0.

    Returns:
        Transition: A dictionary where each key corresponds to the
            respective stacked elements.
    """
    return jax.tree.map(lambda *arr: np.stack(arr, axis=axis), *tree_list)


def concatenate(tree_list: list[Transition], axis=0) -> Transition:
    """Concatenate leaf elements in a pytree (in cardio, most often a list of
    dictionaries) along an existing axis.

    Args:
        tree_list (list[Transition]): List of dictionaries whose leaf
            elements are to be concatenated. Keys in each dictionary
            must match and shapes at each respective key must be the
            same.
        axis (int, optional): The axis in the result array along which
            the input arrays are concatenated. Defaults to 0.

    Returns:
        Transition: A dictionary where each key corresponds to the
            respective concatenated elements.
    """
    return jax.tree.map(lambda *arr: np.concatenate(arr, axis=axis), *tree_list)

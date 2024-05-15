import jax
import numpy as np

def stack(tree_list):
    return jax.tree.map(lambda *arr: np.stack(arr), *tree_list)


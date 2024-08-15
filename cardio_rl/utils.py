import jax.numpy as jnp
import numpy as np
import torch as th


def to_jnp(arr: np.ndarray, dtype=jnp.float32):
    return jnp.asarray(arr, dtype=dtype)


def to_torch(arr: np.ndarray):
    return th.from_numpy(arr)

import jax.numpy as jnp
import numpy as np
import torch as th


def to_np(arr: np.ndarray, dtype=np.float32):
    return arr.astype(dtype)


def to_jnp(arr: np.ndarray, dtype=jnp.float32):
    return jnp.asarray(arr, dtype=dtype)


def to_torch(arr: np.ndarray, dtype=th.float32):
    return th.from_numpy(np.array(arr)).to(dtype)

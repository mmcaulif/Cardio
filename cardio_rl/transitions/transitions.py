import torch as th
import numpy as np
import jax.numpy as jnp


def to_np(arr: np.ndarray):
	return arr.astype(float)

def to_jnp(arr: np.ndarray):
	return jnp.asarray(arr, dtype=jnp.float32)

def to_torch(arr: np.ndarray):
	return th.from_numpy(np.array(arr)).float()
    
import flax.linen as nn

# import jax.numpy as jnp


class MinAtarEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        z = nn.relu(nn.Conv(16, (3, 3), strides=1)(x))
        return z


class NatureEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        z = nn.relu(nn.Conv(32, (8, 8), strides=4)(x))
        z = nn.relu(nn.Conv(64, (4, 4), strides=2)(z))
        z = nn.relu(nn.Conv(64, (3, 3), strides=1)(z))
        return z


class DerEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        z = nn.relu(nn.Conv(32, (5, 5), strides=5)(x))
        z = nn.relu(nn.Conv(64, (5, 5), strides=5)(z))
        return z


class ImpalaEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        z = x
        return z

"""Example of performing sequential Monte Carlo search."""

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp

N_PARTICLES = 16
T = 24
p = 4  # Resampling period


class ValueNet(nn.Module):
    """A simple value network for estimating state values."""

    @nn.compact
    def __call__(self, x):
        """Forward pass."""
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def repeat_state(state, n):
    """Repeat the state for n parallel environments."""
    return jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, axis=0), state)


def act(key, n):
    """Sample n random actions from the environment."""
    actions = jax.random.randint(key, (n,), 0, env.num_actions)
    return actions


key = jax.random.key(0)
key, key_reset, key_step = jax.random.split(key, 3)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("SpaceInvaders-MinAtar")
obs, state = env.reset(key_reset, env_params)
state = repeat_state(state, N_PARTICLES)

# Initialize the weights for each particle.
w = jnp.ones(N_PARTICLES)

# Initialize the environment with the initial state.
for t in range(T):
    # Sample random actions.
    key, key_act = jax.random.split(key)
    actions = act(key_act, N_PARTICLES)

    # Perform the step transition.
    n_obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(None, 0, 0, None))(
        key_step, state, actions, env_params
    )

    # Update weights based on Adv.
    adv = 0.0
    nu = 1.0
    w = w * jnp.exp(adv / nu)

    # Resampling Adaptive Search.
    if t & p == 0:
        w = jnp.ones(N_PARTICLES)


print("<<< here >>>")

"""This code implements a Cross-Entropy Method (CEM) for planning in the Pendulum-v1 environment using JAX and Gymnax."""
import gymnax
import jax
import jax.numpy as jnp

env, env_params = gymnax.make("Pendulum-v1")

key = jax.random.key(0)
key, key_reset, key_step = jax.random.split(key, 3)

I = 20  # Number of iterations
H = 200  # Max horizon length
N = 2048  # Number of particles
A = env.num_actions  # Number of actions
K = 256 # Number of elites to select
ALPHA = 0.1  # Learning rate for CEM


def repeat_tree(tree, n):
    """Repeat each leaf in the tree n times."""
    def _repeat_across(x):
        x = jnp.repeat(x, n, axis=0)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x

    tree = jax.tree_util.tree_map(_repeat_across, tree)
    return tree


def cem_planning(outer_timestep, env_state, env, key, mu):
    """CEM planning algorithm."""
    env_state = repeat_tree(env_state, N)
    upper_bound = env.action_space().high
    lower_bound = env.action_space().low
    init_var = (upper_bound - lower_bound)**2 / 16
    sigma = jnp.ones(mu.shape) * jnp.sqrt(init_var)

    def _cem_ter(carry, _):
        key, mu, sigma = carry
        key, key_step = jax.random.split(key)
        a_cem = mu + jax.random.normal(key_step, (H - outer_timestep, N, A)) * sigma

        def _step(carry, action):
            key, _env_state = carry
            key, key_step = jax.random.split(key)
            _, _env_state, reward, _, _ = jax.vmap(env.step, in_axes=(None, 0, 0, None))(key_step, _env_state, action, env_params)
            return (key, _env_state), reward

        _, reward = jax.lax.scan(_step, (key, env_state), xs=a_cem)
        returns = jnp.sum(reward, axis=0)
        _, indices = jax.lax.top_k(returns, K)
        elites = a_cem[:, indices]

        ### Weighted mean and variance calculation
        topk_returns = returns[indices]
        positive_weights = (topk_returns - jnp.min(topk_returns))
        weight_sum = jnp.sum(positive_weights)
        weights = positive_weights / weight_sum  # Normalize
        weights = weights.reshape(1, -1, 1)  # Add dims for broadcasting
        new_mu = jnp.sum(weights * elites, axis=1, keepdims=True)
        diff_sq = (elites - new_mu) ** 2
        new_sigma = jnp.sqrt(jnp.sum(weights * diff_sq, axis=1, keepdims=True))
        mu = (1.0 - ALPHA) * mu + ALPHA * new_mu
        sigma = (1.0 - ALPHA) * sigma + ALPHA * new_sigma
        carry = key, mu, sigma
        return carry, mu

    (key, mu, _), _ = jax.lax.scan(_cem_ter, (key, mu, sigma), None, length=I)
    return mu[0, 0], mu


obs, env_state = env.reset(key_reset, env_params)
outer_return = 0.0
outer_timestep = 0

mu = jnp.zeros((H, 1, A))

while True:
    action, mu = cem_planning(outer_timestep, env_state, env, key, mu)
    mu = mu[1:]
    obs, env_state, reward, done, _ = env.step(key, env_state, action, env_params)
    outer_return += reward
    outer_timestep += 1
    print(f"Timestep {outer_timestep}, Return so far: {outer_return}, Action: {action}")
    if done:
        break

print(f"Episode finished with return: {outer_return}")

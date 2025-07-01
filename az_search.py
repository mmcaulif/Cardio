"""Example of performing Monte Carlo Tree Search."""

# Gymnax examples: https://github.com/RobertTLange/gymnax/blob/main/examples/00_getting_started.ipynb
# MCTX policy imporvement demo: https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py
# MCTX basic tree search: https://github.com/kenjyoung/mctx_learning_demo/blob/main/basic_tree_search.py

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import mctx


class ValueNet(nn.Module):
    """A simple value network for estimating state values."""

    act_dim: int

    @nn.compact
    def __call__(self, x):
        """Forward pass."""
        x = nn.Conv(16, (3, 3), strides=1)(x)
        x = nn.relu(x)
        x = jnp.ravel(x)

        # Value head
        v = nn.Dense(64)(x)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = nn.tanh(v)

        # Policy head
        p = nn.Dense(64)(x)
        p = nn.relu(p)
        p = nn.Dense(self.act_dim)(p)
        return jnp.squeeze(v), p


def get_search_fn(step_fn, env_params, apply_fn):
    """Returns a function that performs MCTS search on the environment."""

    def recurrent_fn(params, key, actions, env_states):
        key, subkey = jax.random.split(key)
        obs, env_states, rewards, terminals, _ = step_fn(
            subkey, env_states, actions, env_params
        )
        val, pi_logit = apply_fn(params, obs.astype(float))
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=(1.0 - terminals) * 0.99,
            prior_logits=pi_logit,
            value=val,
        )
        return recurrent_fn_output, env_states

    def mcts_search(
        pi_logits, val, state, params, key, temperature=1e-4, num_simulations=1024
    ):
        root = mctx.RootFnOutput(prior_logits=pi_logits, value=val, embedding=state)

        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            temperature=temperature,
        )

        return policy_output.action_weights

    return mcts_search


def main():
    """Main function to run the MCTS example."""
    N_ENVS = 1
    N_SIMS = 16

    # Set random seed for reproducibility.
    key = jax.random.key(0)
    key, key_reset, key_step, key_search, key_act = jax.random.split(key, 5)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("SpaceInvaders-MinAtar")
    reset_fn = jax.vmap(env.reset, in_axes=(0, None))
    step_fn = jax.vmap(env.step, in_axes=(None, 0, 0, None))

    # Vectorise environment functions.
    vmap_keys = jax.random.split(key_reset, N_ENVS)
    obs, state = reset_fn(vmap_keys, env_params)

    # Initialise the neural network.
    net = ValueNet(act_dim=env.num_actions)
    params = net.init(key, obs[0])
    apply_fn = jax.vmap(net.apply, in_axes=(None, 0))

    search_fn = get_search_fn(step_fn, env_params, apply_fn)

    for _ in range(16):
        val, pi_logits = apply_fn(params, obs)
        search_policy = search_fn(
            pi_logits, val, state, params, key_search, num_simulations=N_SIMS
        )
        print("Search policy distribution:", search_policy[0])

        dist = distrax.Categorical(logits=search_policy)
        actions = dist.sample(seed=key_act)

        obs, state, rewards, terminals, _ = step_fn(
            key_step, state, actions, env_params
        )


if __name__ == "__main__":
    main()

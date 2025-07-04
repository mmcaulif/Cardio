"""Example of performing Sequential Monte Carlo search."""

import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

N_PARTICLES = 512
DEPTH = 16
PERIOD = 4  # Resampling period


@chex.dataclass(frozen=True)
class Particle:
    """Particles for the SMC search."""

    weight: jax.Array
    returns: jax.Array
    observation: jax.Array
    state: jax.Array
    init_action: jax.Array
    index: int


def search(env, env_params, n_particles, state, prior_logits, key, apply_fn, params):
    """Perform a search using SMC."""
    key, act_key = jax.random.split(key)
    actions = jax.random.categorical(act_key, prior_logits, shape=(n_particles,))

    n_obs, state, reward, _, _ = jax.vmap(env.step, in_axes=(None, None, 0, None))(
        key, state, actions, env_params
    )

    particles = Particle(
        weight=jnp.ones(n_particles),
        returns=jnp.zeros(n_particles),
        observation=n_obs,
        state=state,
        init_action=actions,
        index=jnp.arange(n_particles, dtype=jnp.int32),
    )

    # TODO: Implement as a jax.scan with a recurrent_fn like in mctx
    for t in range(DEPTH):
        # Sample actions for each particle.
        key, act_key = jax.random.split(key)
        logits = jax.vmap(apply_fn, in_axes=(None, 0))(params, n_obs)
        actions = jax.random.categorical(act_key, logits)

        # Perform the step transition for each particle.
        n_obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(None, 0, 0, None))(
            key, state, actions, env_params
        )

        returns = particles.returns + reward * jnp.pow(0.99, t)

        ### MAIN PROPOSAL, GRPO LIKE ADVANTAGE CALCULATION
        # Liekly to be myopic as has no forward view
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update particles with new observations and weights.
        particles = particles.replace(
            weight=nn.softplus(
                adv
            ),  # Compare with exponential, lots to experiment with
            returns=returns,
            observation=n_obs,
            state=state,
        )

        # print(f"Step {t + 1}/{DEPTH}\n    Indices: {particles.index}\n    Rewards: {reward}\n    Returns: {returns}\n    Advantage: {adv}\n    Weights: {particles.weight}\n    Dist: {nn.softmax(particles.weight)}")

        # Resampling Adaptive Search.
        if t % PERIOD == 0 and t > 0:
            key, resampling_key = jax.random.split(key)
            sampled_idxs = jax.random.categorical(
                resampling_key, particles.weight, shape=(n_particles,)
            )

            particles_resampled = jax.tree_util.tree_map(
                lambda x: x[sampled_idxs], particles
            )

            # Reset weights and returns
            particles = particles_resampled.replace(
                weight=jnp.ones_like(particles.weight)
            )
            # print(f"<<< Resampled particles at step {t}, {particles.index} >>>")

    # print(f"\nRate of collapse: {len(jnp.unique(particles.index))/n_particles * 100}%, final indices: {particles.index}")

    # Figure out how SPO creates the action distribution...
    return particles.init_action


class Policy(nn.Module):
    """Minatar convnet."""

    num_actions: int

    @nn.compact
    def __call__(self, obs):
        """Forward pass."""
        z = nn.Conv(16, (3, 3))(obs)
        z = nn.relu(z)
        z = jnp.ravel(z)
        z = nn.Dense(128)(z)
        z = nn.relu(z)
        logits = nn.Dense(self.num_actions)(z)
        return logits


@jax.grad
def ce_loss(params, apply_fn, obs, labels):
    """Cross entropy loss with integer labels."""
    logits = jax.vmap(apply_fn, in_axes=(None, 0))(params, obs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return -jnp.mean(loss)


def main():
    """Main function to run the SMC search."""
    # TODO: Vectorise the environment during training and vamp the search

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("SpaceInvaders-MinAtar")

    key = jax.random.key(0)
    key, key_init = jax.random.split(key)
    pi = Policy(env.num_actions)
    dummy_obs = env.observation_space(env_params).sample(key_init)

    ts = TrainState.create(
        apply_fn=pi.apply, params=pi.init(key_init, dummy_obs), tx=optax.adam(3e-4)
    )

    T = 0

    for i in range(128):
        key, key_reset, key_step = jax.random.split(key, 3)
        obs, env_state = env.reset(key_reset, env_params)

        obs_seq = []
        reward_seq = []
        action_seq = []

        # Example
        while True:
            T += 1
            key, key_act, key_step = jax.random.split(key, 3)

            prior_logits = ts.apply_fn(ts.params, obs)
            all_actions = search(
                env,
                env_params,
                N_PARTICLES,
                env_state,
                prior_logits,
                key_act,
                ts.apply_fn,
                ts.params,
            )
            logits = nn.one_hot(all_actions, env.num_actions).sum(0)
            action = jax.random.categorical(key_step, logits)

            obs_seq.append(obs)
            action_seq.append(action)

            obs, env_state, reward, done, _ = env.step(
                key_step, env_state, action, env_params
            )
            reward_seq.append(reward)
            if done:
                break

        obs_seq = jnp.array(obs_seq)
        action_seq = jnp.array(action_seq)

        grads = ce_loss(ts.params, ts.apply_fn, obs_seq, action_seq)
        ts = ts.apply_gradients(grads=grads)

        cum_rewards = jnp.array(reward_seq).sum()
        print(f"Episode: {i}\n  Steps: {T}\n  Cumulative reward: {cum_rewards}")


if __name__ == "__main__":
    main()

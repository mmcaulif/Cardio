"""Soft MOE for reinforcement learning from 'Mixtures of Experts Unlock
Parameter Scaling for Deep RL' for discrete environments.

Paper:
Code: https://github.com/google/dopamine/tree/master/dopamine/labs/moes
Hyperparameters:
Experiment details:

__description__

Notes:

To do:
*
"""

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp

from cardio_rl.wrappers import AtariWrapper


class Expert(nn.Module):
    dims: int

    @nn.compact
    def __call__(self, x):
        return nn.relu(nn.Dense(self.dims)(x))


class SoftMOE(nn.Module):
    experts: int = 4
    expert_dim: int = 32
    slots: int = 2

    @nn.compact
    def __call__(self, tokens):
        embedding_d = tokens.shape[-1]

        phi = self.param(
            "phi",
            nn.initializers.lecun_normal(),
            (embedding_d, self.experts, self.slots),
        )

        logits = jnp.einsum("td, dnp -> tnp", tokens, phi)

        dispatch_weights = jax.nn.softmax(logits, axis=1)
        combine_weights = jax.nn.softmax(logits, axis=(-2, -1))

        mixture_inputs = jnp.einsum("td, tnp -> npd", tokens, dispatch_weights)

        Ys = nn.vmap(
            Expert,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.experts,
        )(self.expert_dim)(mixture_inputs)

        Y = jnp.einsum("npd, tnp -> td", Ys, combine_weights)
        return Y


class Q_critic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Conv(32, (8, 8), strides=4)(state))
        z = nn.relu(nn.Conv(64, (4, 4), strides=2)(z))
        z = nn.relu(nn.Conv(64, (3, 3), strides=1)(z))

        z = jnp.reshape(z, [z.shape[0] * z.shape[1], z.shape[2]])
        z = jnp.transpose(z)

        z = SoftMOE()(z)
        z = jnp.reshape(z, (-1))
        q = nn.Dense(self.act_dim)(z)
        return q


def main():
    """Need to figure out the shapes for atari conv nets, its possible you're
    current implementation is very wrong Should the stacked frames be the
    channel??

    By default the shape is [4, 84, 84, 1] which leads to [4, embed_dim]
    as it considers 1 to be the channel. If you squeeze the input it considers input to be [4, 84, 84] with the channel
    dim being 84...
    """

    env = gym.make("QbertNoFrameskip-v4")
    env = AtariWrapper(env)

    s, _ = env.reset()

    print(s.shape)

    module = Q_critic(env.action_space.n)
    params = module.init(jax.random.PRNGKey(0), s)

    output = jax.vmap(module.apply, in_axes=(None, 0))(params, jnp.expand_dims(s, 0))
    print(output.shape)

    # batch_tokens = jnp.ones((B, T, D))
    # output = jax.vmap(module.apply, in_axes=(None, 0))(params, batch_tokens)

    # B = 16
    # T = 8
    # D = 128

    # dummy_tokens = jnp.ones((T, D))

    # module = SoftMOE(
    #     experts=4,
    #     embedding_d=D,
    #     slots=2
    # )

    # params = module.init(jax.random.PRNGKey(0), dummy_tokens)

    # batch_tokens = jnp.ones((B, T, D))
    # output = jax.vmap(module.apply, in_axes=(None, 0))(params, batch_tokens)

    # print(output.shape)


if __name__ == "__main__":
    main()

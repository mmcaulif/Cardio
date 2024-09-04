"""Data Efficient Rainbow from 'When to use parametric models in reinforcement
learning?' for discrete environments (Atari 100k).

Paper:
Hyperparameters: https://github.com/google/dopamine/blob/master/dopamine/labs/atari_100k/configs/DER.gin
Experiment details:

Rainbow with tuned hyperparameters for sample efficiency

Notes:

To do:
* Noisy nets
* Benchmarking (Atari 100k)
"""

from typing import NamedTuple, Optional

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp

import cardio_rl as crl
from examples.jax.rainbow import Rainbow


class NetworkOutputs(NamedTuple):
    q_values: jnp.ndarray
    q_logits: jnp.ndarray


class Q_critic(nn.Module):
    action_dim: int
    support: jnp.ndarray

    @nn.compact
    def __call__(self, state):
        n_atoms = len(self.support)

        z = nn.relu(nn.Dense(128)(state))
        z = nn.relu(nn.Dense(128)(z))
        v = nn.Dense(n_atoms)(z)
        a = nn.Dense(self.action_dim * n_atoms)(z)

        v = jnp.expand_dims(v, -2)
        a = jnp.reshape(a, (self.action_dim, n_atoms))
        q_logits = v + a - a.mean(-2, keepdims=True)

        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * self.support, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)

        return NetworkOutputs(q_values=q_values, q_logits=q_logits)


class DER(Rainbow):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 2_000,
        n_steps: int = 10,
        v_max: float = 10,
        v_min: Optional[float] = None,
        n_atoms: int = 51,
        optim_kwargs: dict = {"learning_rate": 1e-4, "eps": 0.00015},
        init_eps: float = 1,
        min_eps: float = 0.01,
        schedule_len: int = 2000,
        use_rmsprop: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            env,
            critic,
            gamma,
            targ_freq,
            n_steps,
            v_max,
            v_min,
            n_atoms,
            optim_kwargs,
            init_eps,
            min_eps,
            schedule_len,
            use_rmsprop,
            seed,
        )


def main():
    env = gym.make("LunarLander-v2")

    agent = (
        DER(
            env,
            Q_critic(action_dim=2, support=jnp.linspace(-500, 250, 51)),
            schedule_len=100_000,
            v_max=250,
            v_min=-500,
        ),
    )

    runner = crl.OffPolicyRunner(
        env,
        agent=agent,
        buffer=crl.buffers.PrioritisedBuffer(env, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
    )

    runner.run(98_400)


if __name__ == "__main__":
    main()

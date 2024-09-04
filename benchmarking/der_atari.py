from typing import NamedTuple

import flax.linen as nn
import gymnasium as gym
import jax.debug
import jax.numpy as jnp

import cardio_rl as crl
from cardio_rl.wrappers import AtariWrapper
from examples.intermediate.der import DER

# https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/rainbow/agent.py
# https://github.com/google/dopamine/blob/master/dopamine/jax/agents/full_rainbow/full_rainbow_agent.py
"""
Every so often I get:
/home/manus/github/Cardio/cardio_rl/buffers/prioritised_buffer.py:103: RuntimeWarning: invalid value encountered in divide
  probs = self.sumtree.data[sample_indxs] / self.sumtree.total

Is sumtree.total=0 ???

^think the above is solved after fixing the per buffer

Also get random crashes around ~40,000 environment steps with envpool
"""


class NetworkOutputs(NamedTuple):
    q_values: jnp.ndarray
    q_logits: jnp.ndarray


class Q_critic(nn.Module):
    act_dim: int
    support: jnp.ndarray

    @nn.compact
    def __call__(self, state):
        n_atoms = len(self.support)

        z = nn.relu(nn.Conv(32, (5, 5), strides=5)(state))
        z = nn.relu(nn.Conv(64, (5, 5), strides=5)(z))
        z = jnp.reshape(z, (-1))

        z = nn.relu(nn.Dense(256)(z))
        v = nn.Dense(n_atoms)(z)
        a = nn.Dense(self.act_dim * n_atoms)(z)

        v = jnp.expand_dims(v, -2)
        a = jnp.reshape(a, (self.act_dim, n_atoms))
        q_logits = v + a - a.mean(-2, keepdims=True)

        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * self.support, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)

        return NetworkOutputs(q_values=q_values, q_logits=q_logits)


def main():
    """
    Amidar:
        random: 5.8, DER: 188.6
    Qbert:
        random: 163.9, DER: 1152.9
    """

    env = gym.make("QbertNoFrameskip-v4")
    env = AtariWrapper(env)

    eval_env = gym.make("QbertNoFrameskip-v4")
    eval_env = AtariWrapper(eval_env, eval=True)

    agent = DER(
        env=env,
        critic=Q_critic(
            act_dim=env.action_space.n, support=jnp.linspace(-10.0, 10.0, 51)
        ),
    )

    runner = crl.OffPolicyRunner(
        env=env,
        agent=agent,
        buffer=crl.buffers.PrioritisedBuffer(env=env, capacity=100_000, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
        eval_env=eval_env,
    )

    def how_many_rollouts(env_steps: int, warmup_len: int, rollout_len: int) -> int:
        return int((env_steps - warmup_len) / rollout_len)

    steps = how_many_rollouts(100_000, runner.warmup_len, runner.rollout_len)

    runner.run(rollouts=steps, eval_freq=15_000, eval_episodes=10)


if __name__ == "__main__":
    main()

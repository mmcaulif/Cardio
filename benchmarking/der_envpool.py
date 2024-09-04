from typing import NamedTuple

import envpool
import flax.linen as nn
import jax.debug
import jax.numpy as jnp
from tqdm import trange

import cardio_rl as crl
from cardio_rl.wrappers import EnvPoolWrapper
from examples.intermediate.der import DER

# https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/rainbow/agent.py
# https://github.com/google/dopamine/blob/master/dopamine/jax/agents/full_rainbow/full_rainbow_agent.py
"""
Every so often I get:
/home/manus/github/Cardio/cardio_rl/buffers/prioritised_buffer.py:103: RuntimeWarning: invalid value encountered in divide
  probs = self.sumtree.data[sample_indxs] / self.sumtree.total

Is sumtree.total=0 ???

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
    env = envpool.make_gymnasium(
        "Qbert-v5", num_envs=1, episodic_life=True, reward_clip=True
    )
    env = EnvPoolWrapper(env)

    eval_env = envpool.make_gymnasium("Qbert-v5", num_envs=1)
    eval_env = EnvPoolWrapper(eval_env)

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

    # runner.run(rollouts=98_400, eval_freq=15_000, eval_episodes=10)
    # return

    import tracemalloc

    from tqdm.contrib.logging import logging_redirect_tqdm

    tracemalloc.start()

    rollouts = 98_400
    eval_freq = 5_000

    for t in trange(rollouts):
        data = runner.step()
        updated_data = runner.agent.update(data)  # type: ignore
        if updated_data:
            runner.update(updated_data)
        if t % eval_freq == 0 and t > 0:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")
            for stat in top_stats[:5]:
                with logging_redirect_tqdm():
                    print(stat)


if __name__ == "__main__":
    main()

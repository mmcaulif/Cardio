import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp

import cardio_rl as crl
from cardio_rl.wrappers import AtariWrapper
from examples.jax.drq_e import DrQ


class Q_critic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z = crl.nn.DerEncoder()(state)  # Which is used for DrQ?
        z = jnp.reshape(z, (-1))

        z = nn.relu(nn.Dense(256)(z))
        v = nn.Dense(1)(z)
        a = nn.Dense(self.act_dim)(z)

        v = jnp.expand_dims(v, -2)
        q = v + a - a.mean(-1, keepdims=True)
        return q.squeeze()


def main():
    """
    Qbert:
        random: 163.9, DER: 1152.9, DrQ:
    """

    env = gym.make("QbertNoFrameskip-v4")
    env = AtariWrapper(env)

    eval_env = gym.make("QbertNoFrameskip-v4")
    eval_env = AtariWrapper(eval_env, eval=True)

    agent = DrQ(
        env=env,
        critic=Q_critic(act_dim=env.action_space.n),
    )

    runner = crl.OffPolicyRunner(
        env=env,
        agent=agent,
        buffer=crl.buffers.TreeBuffer(env=env, capacity=100_000, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
        eval_env=eval_env,
    )

    def how_many_rollouts(env_steps: int, warmup_len: int, rollout_len: int) -> int:
        return int((env_steps - warmup_len) / rollout_len)

    steps = how_many_rollouts(100_000, runner.warmup_len, runner.rollout_len)

    runner.run(rollouts=steps, eval_freq=7_500, eval_episodes=50)


if __name__ == "__main__":
    main()
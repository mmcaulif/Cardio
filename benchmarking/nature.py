import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp

import cardio_rl as crl
from cardio_rl.wrappers import AtariWrapper
from examples.jax.dqn import DQN


class Q_critic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z = crl.nn.NatureEncoder(state)
        z = jnp.reshape(z, (-1))
        z = nn.relu(nn.Dense(512)(z))
        q = nn.Dense(self.act_dim)(z)
        return q


def main():
    env = gym.make("QbertNoFrameskip-v4")
    env = AtariWrapper(env, action_repeat_probability=0.25)

    eval_env = gym.make("QbertNoFrameskip-v4")
    eval_env = AtariWrapper(eval_env, action_repeat_probability=0.25, eval=True)

    agent = DQN(
        env=env,
        critic=Q_critic(
            act_dim=env.action_space.n,
        ),
        targ_freq=10_000,
        init_eps=1.0,
        min_eps=0.1,
        schedule_len=1_000_000,
    )

    runner = crl.OffPolicyRunner(
        env=env,
        agent=agent,
        batch_size=32,
        warmup_len=100,
        eval_env=eval_env,
    )

    def how_many_rollouts(env_steps: int, warmup_len: int, rollout_len: int) -> int:
        return int((env_steps - warmup_len) / rollout_len)

    steps = how_many_rollouts(10_000_000, runner.warmup_len, runner.rollout_len)

    runner.run(rollouts=steps, eval_freq=100_000, eval_episodes=50)


if __name__ == "__main__":
    main()

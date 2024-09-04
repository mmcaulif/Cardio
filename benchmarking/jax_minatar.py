import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp

import cardio_rl as crl
from examples.jax.dqn import DQN


class Q_critic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Conv(16, (3, 3), strides=1)(state))
        z = jnp.reshape(z, -1)
        z = nn.relu(nn.Dense(128)(z))
        q = nn.Dense(self.act_dim)(z)
        return q


def main():
    env = gym.make("MinAtar/Freeway-v1")
    # env = gym.make("MinAtar/SpaceInvaders-v1")

    agent = DQN(
        env=env,
        critic=Q_critic(act_dim=env.action_space.n),
        optim_kwargs={
            "learning_rate": 0.00025,
            "decay": 0.95,
            "centered": True,
            "eps": (0.01) ** 2,
            # approximates the pytorch implementation epsilon value: https://github.com/google-deepmind/optax/issues/532
        },
        init_eps=1.0,
        min_eps=0.1,
        schedule_len=100_000,
        use_rmsprop=True,
    )

    runner = crl.OffPolicyRunner(
        env=env,
        agent=agent,
        capacity=100_000,
        rollout_len=1,
        batch_size=32,
        warmup_len=5_000,
    )

    def how_many_rollouts(env_steps: int, warmup_len: int, rollout_len: int) -> int:
        return int((env_steps - warmup_len) / rollout_len)

    steps = how_many_rollouts(600_000, runner.warmup_len, runner.rollout_len)

    runner.run(rollouts=steps, eval_freq=25_000, eval_episodes=20)


if __name__ == "__main__":
    main()

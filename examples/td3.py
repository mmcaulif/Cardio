from functools import partial
import logging
from typing import Any, Optional
import gymnasium as gym
import jax
import numpy as np

import flax.linen as nn
import jax.numpy as jnp
import optax  # type: ignore
from flax.training.train_state import TrainState
import rlax  # type: ignore

import cardio_rl as crl  # type: ignore


class TargetTrainState(TrainState):
    target_params: Any


class Critic(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        sa = jnp.concat([state, action], axis=-1)

        z_1 = nn.relu(nn.Dense(400)(sa))
        z_1 = nn.relu(nn.Dense(300)(z_1))
        q1 = nn.Dense(1)(z_1)

        z_2 = nn.relu(nn.Dense(400)(sa))
        z_2 = nn.relu(nn.Dense(300)(z_2))
        q2 = nn.Dense(1)(z_2)

        return q1, q2


class Actor(nn.Module):
    act_dim: int
    scale: float = 1.0

    @nn.compact
    def __call__(self, state):
        z_pi = nn.tanh(nn.Dense(400)(state))
        z_pi = nn.tanh(nn.Dense(300)(z_pi))
        logits = nn.tanh(nn.Dense(self.act_dim)(z_pi)) * self.scale
        return logits


class TD3(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        optim_kwargs: dict = {"learning_rate": 1e-3},
        seed: Optional[int] = None,
    ):
        self.seed = np.random.randint(0, int(2e16)) if seed is None else seed
        logging.info(f"Seed: {self.seed}")
        self.key = jax.random.PRNGKey(self.seed)

        self.env = env

        self.gamma = gamma

        actor = Actor(act_dim=1, scale=2.0)
        critic = Critic()

        s_dummy = jnp.zeros(env.observation_space.shape)
        a_dummy = jnp.zeros([1])

        self.key, actor_key, critic_key = jax.random.split(self.key, 3)

        self.actor_ts = TargetTrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, s_dummy),
            target_params=actor.init(actor_key, s_dummy),
            tx=optax.adam(**optim_kwargs),
        )

        self.critic_ts = TargetTrainState.create(
            apply_fn=critic.apply,
            params=critic.init(critic_key, s_dummy, a_dummy),
            target_params=critic.init(critic_key, s_dummy, a_dummy),
            tx=optax.adam(**optim_kwargs),
        )

        self.update_count = 0

        def _step(train_state: TrainState, state: np.ndarray, key):
            action = train_state.apply_fn(train_state.params, state)
            noise = jax.random.normal(key, action.shape) * 0.2
            noise = jnp.clip(noise, min=-0.5, max=0.5)
            action = jnp.clip(action + noise, min=-2.0, max=2.0)
            return action

        self._step = jax.jit(_step)
        self._eval_step = jax.jit(actor.apply)

    def update(self, batch):
        s, a, r, s_p, d = batch["s"], batch["a"], batch["r"], batch["s_p"], batch["d"]

        @jax.grad
        def loss(params, critic_train_state, actor_train_state, key):
            a_p = actor_train_state.apply_fn(actor_train_state.params, s_p)
            noise = jax.random.normal(key, a_p.shape) * 0.2
            noise = jnp.clip(noise, min=-0.5, max=0.5)
            a_p = jnp.clip(a_p + noise, min=-2.0, max=2.0)

            q_p1, qp2 = critic_train_state.apply_fn(
                critic_train_state.target_params, s_p, a_p
            )

            q_p = jnp.minimum(q_p1, qp2)

            y = r + 0.99 * q_p * ~d

            q1, q2 = critic_train_state.apply_fn(params, s, a)
            c_loss = jnp.mean(rlax.l2_loss(q1 - y)) + jnp.mean(rlax.l2_loss(q2 - y))
            return c_loss

        self.key, update_key = jax.random.split(self.key)

        c_grads = loss(self.critic_ts.params, self.critic_ts, self.actor_ts, update_key)

        self.critic_ts = self.critic_ts.apply_gradients(grads=c_grads)

        self.update_count += 1
        if self.update_count % 2 == 0:
            pass
            # q1, q2 = self.critic(s, self.actor(s_p))
            # policy_loss = -((q1 + q2) * 0.5).mean()

            # self.a_optimizer.zero_grad()
            # policy_loss.backward()
            # self.a_optimizer.step()

        return {}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.actor_ts, state, act_key)
        return np.asarray(action), {}

    def eval_step(self, state):
        action = self._eval_step(self.actor_ts.params, state)
        return np.asarray(action)


def main():
    env = gym.make("Pendulum-v1")
    runner = crl.Runner.off_policy(
        env=env,
        agent=TD3(env),
        buffer_kwargs={"batch_size": 256},
        rollout_len=1,
    )
    runner.run(rollouts=190_000)


if __name__ == "__main__":
    main()

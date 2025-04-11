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


def _step(train_state: TrainState, state: np.ndarray, key):
    action = train_state.apply_fn(train_state.params, state)
    noise = jax.random.normal(key, action.shape) * 0.2
    noise = jnp.clip(noise, min=-0.5, max=0.5)
    action = jnp.clip(action + noise, min=-2.0, max=2.0)
    return action


def _update(critic_ts, actor_ts, s, a, r, s_p, d, gamma, t, key):
    def critic_loss_fn(params, critic_ts, actor_ts, key):
        a_p = actor_ts.apply_fn(
            actor_ts.params, s_p
        )  # TODO: does TD3 use actor target parameters?
        noise = jax.random.normal(key, a_p.shape) * 0.2
        noise = jnp.clip(noise, min=-0.5, max=0.5)
        a_p = jnp.clip(a_p + noise, min=-2.0, max=2.0)

        q_p1, qp2 = critic_ts.apply_fn(critic_ts.target_params, s_p, a_p)

        q_p = jnp.minimum(q_p1, qp2)

        y = r + gamma * q_p * (1.0 - d)

        q1, q2 = critic_ts.apply_fn(params, s, a)
        loss = jnp.mean(rlax.l2_loss(q1 - y)) + jnp.mean(rlax.l2_loss(q2 - y))
        metrics = {
            "Critic loss": loss,
        }
        return loss, metrics

    def actor_loss_fn(params, critic_ts, actor_ts):
        a = actor_ts.apply_fn(params, s)
        q1, _ = critic_ts.apply_fn(critic_ts.params, s, a)
        loss = -jnp.mean(q1)
        metrics = {
            "Policy loss": loss,
        }
        return loss, metrics

    key, update_key = jax.random.split(key)
    c_grads, metrics = jax.grad(critic_loss_fn, has_aux=True)(
        critic_ts.params, critic_ts, actor_ts, update_key
    )
    new_critic_ts = critic_ts.apply_gradients(grads=c_grads)

    a_grads, policy_metrics = jax.grad(actor_loss_fn, has_aux=True)(
        actor_ts.params, critic_ts, actor_ts
    )
    new_actor_ts = actor_ts.apply_gradients(grads=a_grads)

    new_actor_ts = jax.lax.cond(
        jnp.mod(t, 2) == 0, lambda _: new_actor_ts, lambda _: actor_ts, None
    )

    new_critic_ts = jax.lax.cond(
        jnp.mod(t, 2) == 0,
        lambda _: new_critic_ts.replace(
            target_params=optax.incremental_update(
                new_critic_ts.params, new_critic_ts.target_params, 0.005
            )
        ),
        lambda _: new_critic_ts,
        None,
    )

    metrics.update(policy_metrics)

    return new_critic_ts, new_actor_ts, key, metrics


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

        self._step = jax.jit(_step)
        self._eval_step = jax.jit(actor.apply)
        self._update = jax.jit(partial(_update, gamma=gamma))

    def update(self, batches):
        self.critic_ts, self.actor_ts, self.key, metrics = self._update(
            self.critic_ts,
            self.actor_ts,
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
            t=self.critic_ts.step,
            key=self.key,
        )
        self.update_count += 1
        return metrics, {}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.actor_ts, state, act_key)
        return np.asarray(action), {}

    def eval_step(self, state):
        action = self._eval_step(self.actor_ts.params, state)
        return np.asarray(action)


def main():
    np.random.seed(42)

    env = gym.make("Pendulum-v1")
    runner = crl.Runner.off_policy(
        env=env,
        agent=TD3(env, gamma=0.98),
        buffer_kwargs={"batch_size": 256},
        rollout_len=1,
    )
    runner.run(rollouts=190_000)


if __name__ == "__main__":
    main()

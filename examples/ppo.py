from functools import partial
import logging
from typing import Optional
import gymnasium as gym
import jax
import numpy as np

import distrax  # type: ignore
import flax.linen as nn
import jax.numpy as jnp
import optax  # type: ignore
from flax.training.train_state import TrainState

import cardio_rl as crl  # type: ignore
from cardio_rl.types import Transition


class Actor(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z_pi = nn.tanh(nn.Dense(64)(state))
        z_pi = nn.tanh(nn.Dense(64)(z_pi))
        logits = nn.Dense(self.act_dim)(z_pi)
        return logits


class Critic(nn.Module):
    @nn.compact
    def __call__(self, state):
        z_v = nn.tanh(nn.Dense(64)(state))
        z_v = nn.tanh(nn.Dense(64)(z_v))
        value = nn.Dense(1)(z_v)
        return value.squeeze(-1)


class PPO(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epochs: int = 10,
        batch_size: int = 64,
        ent_coeff: float = 0.0,
        max_grad_norm: float = 0.5,
        optimiser: optax.GradientTransformation = optax.adam,
        optim_kwargs: dict = {"learning_rate": 3e-4},
        seed: Optional[int] = None,
    ):
        self.seed = np.random.randint(0, int(2e16)) if seed is None else seed
        logging.info(f"Seed: {self.seed}")
        self.key = jax.random.PRNGKey(self.seed)

        self.env = env

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coeff = ent_coeff

        actor = Actor(act_dim=2)
        critic = Critic()

        dummy = jnp.zeros(env.observation_space.shape)

        self.key, actor_key, critic_key = jax.random.split(self.key, 3)

        self.actor_ts = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, dummy),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optimiser(**optim_kwargs)
            ),
        )

        self.critic_ts = TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(critic_key, dummy),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(**optim_kwargs)
            ),
        )

        def _step(train_state: TrainState, state: np.ndarray, key):
            logits = train_state.apply_fn(train_state.params, state)
            dist = distrax.Categorical(logits)
            action = dist.sample(seed=key)
            log_prob = dist.log_prob(action)
            return action, log_prob

        self._step = jax.jit(_step)
        self._eval_step = jax.jit(actor.apply)

    @staticmethod
    @partial(jax.jit, static_argnames=["ent_coeff"])
    def _update(
        actor_ts: TrainState,
        critic_ts: TrainState,
        s,
        a,
        returns,
        adv,
        old_log_probs,
        ent_coeff,
    ):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # Normalise adv

        def actor_loss(params):
            logits = actor_ts.apply_fn(params, s)

            dist = distrax.Categorical(logits)
            log_probs = dist.log_prob(a)
            entropy = dist.entropy()

            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_probs - jax.lax.stop_gradient(old_log_probs))

            # clipped surrogate loss
            policy_loss_1 = adv * ratio
            policy_loss_2 = adv * jnp.clip(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            entropy_loss = -jnp.mean(entropy)

            loss = policy_loss + ent_coeff * entropy_loss
            return loss

        grads = jax.grad(actor_loss)(actor_ts.params)
        actor_ts = actor_ts.apply_gradients(grads=grads)

        def critic_loss(params):
            v = critic_ts.apply_fn(params, s)
            return ((returns - v) ** 2).mean()

        grads = jax.grad(critic_loss)(critic_ts.params)
        critic_ts = critic_ts.apply_gradients(grads=grads)

        return actor_ts, critic_ts

    def update(self, batches):
        v = self.critic_ts.apply_fn(self.critic_ts.params, batches["s"])
        last_values = self.critic_ts.apply_fn(self.critic_ts.params, batches["s_p"][-1])

        last_gae_lam = 0
        gae = np.zeros_like(v)

        for step in reversed(range(2048)):
            if step == 2048 - 1:
                next_non_terminal = 1.0 - batches["d"][-1]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - batches["d"][step + 1]
                next_values = v[step + 1]
            delta = (
                batches["r"][step]
                + self.gamma * next_values * next_non_terminal
                - v[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            gae[step] = last_gae_lam

        returns = gae + v

        batches.update({"gae": gae, "returns": returns})

        def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
            shape = arr.shape
            if len(shape) < 3:
                return arr.reshape(shape[0] * shape[1])
            else:
                return arr.reshape(shape[0] * shape[1], *shape[2:])

        batches = jax.tree.map(swap_and_flatten, batches)

        idxs = np.arange(len(batches["s"]))
        num_mb = len(batches["s"]) // self.batch_size

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            start_idx = 0
            for n in range(num_mb):
                mb_idxs = idxs[start_idx : start_idx + self.batch_size]
                self.actor_ts, self.critic_ts = self._update(
                    self.actor_ts,
                    self.critic_ts,
                    batches["s"][mb_idxs],
                    batches["a"][mb_idxs],
                    batches["returns"][mb_idxs],
                    batches["gae"][mb_idxs],
                    batches["log_prob"][mb_idxs],
                    self.ent_coeff,
                )
                start_idx += self.batch_size

        return {}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action, log_prob = self._step(self.actor_ts, state, act_key)
        return np.asarray(action), {"log_prob": log_prob}

    def eval_step(self, state):
        logits = self._eval_step(self.actor_ts.params, state)
        action = logits.argmax(-1)
        return np.asarray(action)


def main():
    env_fns = [lambda: gym.make("CartPole-v1")] * 16
    envs = gym.vector.SyncVectorEnv(env_fns)

    eval_env = gym.make("CartPole-v1")

    runner = crl.Runner.on_policy(
        env=envs,
        agent=PPO(envs),
        rollout_len=2048,
        eval_env=eval_env,
    )

    runner.run(128, eval_freq=4, eval_episodes=50)


if __name__ == "__main__":
    main()

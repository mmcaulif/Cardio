import copy
import logging
from functools import partial
from typing import Optional

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from flax.training.train_state import TrainState

import cardio_rl as crl


def _step(ts: TrainState, state: np.ndarray, epsilon, key):
    q_values = ts.apply_fn(ts.params, state)
    action = distrax.EpsilonGreedy(q_values, epsilon).sample(seed=key)
    return action


def _update(ts: TrainState, targ_params, s, a, r, s_p, d, gamma):
    def loss_fn(params, apply_fn, s, a, r, s_p, d):
        q = jax.vmap(apply_fn, in_axes=(None, 0))(params, s)
        q_p = jax.vmap(apply_fn, in_axes=(None, 0))(targ_params, s_p)
        discount = gamma * (1 - d)
        error = jax.vmap(rlax.q_learning)(q, a, r, discount, q_p)
        mse = jnp.mean(rlax.l2_loss(error))
        metrics = {
            "Loss": mse,
            "Td-error mean": jnp.mean(error),
            "Td-error std": jnp.std(error),
        }
        return mse, metrics

    a = jnp.squeeze(a, -1)
    r = jnp.squeeze(r, -1)
    d = jnp.squeeze(d, -1)
    grads, metrics = jax.grad(loss_fn, has_aux=True)(
        ts.params, ts.apply_fn, s, a, r, s_p, d
    )
    new_ts = ts.apply_gradients(grads=grads)
    return new_ts, metrics


class Q_critic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Dense(64)(state))
        z = nn.relu(nn.Dense(64)(z))
        q = nn.Dense(self.act_dim)(z)
        return q


class DQN(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"learning_rate": 1e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
        use_rmsprop: bool = False,
        seed: Optional[int] = None,
    ):
        seed = np.random.randint(0, int(2e16)) if seed is None else seed
        logging.info(f"Seed: {seed}")
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        dummy = env.observation_space.sample()
        critic = Q_critic(act_dim=env.action_space.n)
        params = critic.init(init_key, dummy)
        self.targ_params = copy.deepcopy(params)
        self.targ_freq = targ_freq

        if use_rmsprop:
            optimizer = optax.rmsprop(**optim_kwargs)
        else:
            optimizer = optax.adam(**optim_kwargs)

        self.ts = TrainState.create(apply_fn=critic.apply, params=params, tx=optimizer)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

        self._step = jax.jit(_step)
        self._update = jax.jit(partial(_update, gamma=gamma))

    def update(self, batches):
        self.ts, metrics = self._update(
            self.ts,
            self.targ_params,
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
        )

        if self.ts.step % self.targ_freq == 0:
            self.targ_params = self.ts.params

        return metrics, {}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.ts, state, self.eps, act_key)
        action = np.asarray(action)
        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action.squeeze(), {}

    def eval_step(self, state: np.ndarray):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.ts, state, 0.001, act_key)
        action = np.asarray(action)
        return action.squeeze()


def main():
    SEED = 42

    np.random.seed(SEED)

    env = gym.make("CartPole-v1")

    runner = crl.Runner.off_policy(
        env=env,
        agent=DQN(env, seed=SEED),
        buffer_kwargs={"batch_size": 32},
        rollout_len=4,
    )
    runner.run(rollouts=122_500, eval_freq=12_500)


if __name__ == "__main__":
    main()

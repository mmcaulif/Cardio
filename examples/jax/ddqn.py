import logging
from typing import Optional

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from flax.training.train_state import TrainState

import cardio_rl as crl


class Q_critic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Dense(64)(state))
        z = nn.relu(nn.Dense(64)(z))
        q = nn.Dense(self.action_dim)(z)
        return q


class DDQN(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"learning_rate": 1e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
        use_rmsprop: bool = False,
        seed: Optional[int] = None,
    ):
        seed = seed or np.random.randint(0, 2e16)
        logging.info(f"Seed: {seed}")
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        self.env = env

        dummy = jnp.zeros(env.observation_space.shape)
        params = critic.init(init_key, dummy)
        self.targ_params = critic.init(init_key, dummy)

        if use_rmsprop:
            optimizer = optax.rmsprop(**optim_kwargs)
        else:
            optimizer = optax.adam(**optim_kwargs)

        self.ts = TrainState.create(apply_fn=critic.apply, params=params, tx=optimizer)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

        def _step(train_state, state):
            q = train_state.apply_fn(train_state.params, state)
            action = jnp.argmax(q, axis=-1)
            return action

        self._step = jax.jit(_step)

        def _update(train_state: TrainState, targ_params, s, a, r, s_p, d):
            def loss_fn(params, train_state: TrainState, s, a, r, s_p, d):
                q = train_state.apply_fn(params, s)
                q_p_value = train_state.apply_fn(targ_params, s_p)
                q_p_selector = train_state.apply_fn(params, s_p)
                discount = gamma * (1 - d)
                error = jax.vmap(rlax.double_q_learning)(
                    q, a, r, discount, q_p_value, q_p_selector
                )
                mse = jnp.mean(jnp.square(error))
                return mse

            a = jnp.squeeze(a, -1)
            r = jnp.squeeze(r, -1)
            d = jnp.squeeze(d, -1)
            grads = jax.grad(loss_fn)(train_state.params, train_state, s, a, r, s_p, d)
            train_state = train_state.apply_gradients(grads=grads)

            targ_params = optax.periodic_update(
                train_state.params,
                targ_params,
                train_state.step,
                targ_freq,
            )
            return train_state, targ_params

        self._update = jax.jit(_update)

    def update(self, batches):
        s, a, r, s_p, d = (
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
        )
        self.ts, self.targ_params = self._update(
            self.ts,
            self.targ_params,
            s,
            a,
            r,
            s_p,
            d,
        )
        return {}

    def step(self, state):
        # self.key, act_key = jax.random.split(self.key)
        # if jax.random.uniform(act_key) > self.eps:
        if np.random.rand() > self.eps:
            action = self._step(self.ts, state)
            action = np.asarray(action)
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}

    def eval_step(self, state: np.ndarray):
        action = self._step(self.ts, state)
        action = np.asarray(action)
        return action


def main():
    env = gym.make("CartPole-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=DDQN(env, Q_critic(action_dim=2)),
        rollout_len=4,
        batch_size=32,
    )
    runner.run(rollouts=12_500, eval_freq=1_250)


if __name__ == "__main__":
    main()

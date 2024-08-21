import logging
from typing import Optional

import flax.linen as nn
import gymnasium as gym
import jax
import jax.debug
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from flax.training.train_state import TrainState

import cardio_rl as crl


class Q_critic(nn.Module):
    action_dim: int
    n_atoms: int = 51

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Dense(128)(state))
        z = nn.relu(nn.Dense(128)(z))
        v = nn.Dense(self.n_atoms)(z)
        a = nn.Dense(self.action_dim * self.n_atoms)(z)

        v = jnp.expand_dims(v, -2)
        a = jnp.reshape(a, (self.action_dim, self.n_atoms))
        q = v + a - a.mean(-2, keepdims=True)
        return q


class DQN(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"learning_rate": 3e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
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

        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(**optim_kwargs)
        )

        self.ts = TrainState.create(apply_fn=critic.apply, params=params, tx=optimizer)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

        atoms = jnp.expand_dims(jnp.linspace(0, 200, 51), 0)

        def _step(train_state: TrainState, state: np.ndarray):
            q_logits = train_state.apply_fn(train_state.params, state)
            dist = nn.softmax(q_logits, axis=-1)
            q = (dist * atoms).sum(-1)
            action = jnp.argmax(q, axis=-1)
            return action

        self._step = jax.jit(_step)

        def _update(train_state: TrainState, targ_params, s, a, r, s_p, d, w):
            def loss_fn(params, train_state: TrainState, s, a, r, s_p, d, w):
                q_logits = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(params, s)
                q_p_logits = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(
                    targ_params, s_p
                )
                discount = jnp.power(gamma, 3) * (1 - d)

                _atoms = jnp.repeat(atoms, 32, axis=0)

                # TODO: try find an example of this function being used and make sure you're using it right
                error = jax.vmap(rlax.categorical_q_learning)(
                    _atoms, q_logits, a, r, discount, _atoms, q_p_logits
                )
                mse = jnp.mean(jnp.square(error) * w)
                return mse, error

            grads, error = jax.grad(loss_fn, has_aux=True)(
                train_state.params, train_state, s, a, r, s_p, d, w
            )
            train_state = train_state.apply_gradients(grads=grads)

            targ_params = optax.periodic_update(
                train_state.params,
                targ_params,
                train_state.step,
                targ_freq,
            )
            return train_state, targ_params, error

        self._update = jax.jit(_update)

    def update(self, batches):
        s, a, r, s_p, d, w = (
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
            batches["w"],
        )

        # TODO: turn to scan and vmap
        returns = np.zeros(r.shape[0])
        for i in reversed(range(r.shape[1])):
            returns += 0.99 * r[:, i]

        r = returns

        a = jnp.squeeze(a, -1)
        d = jnp.squeeze(d, -1)

        self.ts, self.targ_params, error = self._update(
            self.ts, self.targ_params, s, a, r, s_p, d, w
        )
        return {"idxs": batches["idxs"], "p": np.asarray(jnp.abs(error))}

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
        if np.random.rand() > 0.001:
            action = self._step(self.ts, state)
            action = np.asarray(action).squeeze(0)
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action


def main():
    env = gym.make("CartPole-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=DQN(env, Q_critic(action_dim=2), schedule_len=15_000),
        buffer=crl.buffers.PrioritisedBuffer(env, n_steps=3),
        rollout_len=4,
        batch_size=32,
        n_step=3,
    )
    runner.run(rollouts=12_500, eval_freq=1_250)


if __name__ == "__main__":
    main()

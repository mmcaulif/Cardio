import copy
import logging
from typing import NamedTuple, Optional

import distrax  # type: ignore
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import rlax  # type: ignore
from flax.training.train_state import TrainState

import cardio_rl as crl


class NetworkOutputs(NamedTuple):
    q_values: jnp.ndarray
    q_logits: jnp.ndarray


class Q_critic(nn.Module):
    action_dim: int
    support: jnp.ndarray

    @nn.compact
    def __call__(self, state, key, eval=False):
        n_atoms = len(self.support)

        z = nn.relu(crl.nn.NoisyDense(128)(state, key, eval))
        z = nn.relu(crl.nn.NoisyDense(128)(z, key, eval))
        v = crl.nn.NoisyDense(n_atoms)(z, key, eval)
        a = crl.nn.NoisyDense(self.action_dim * n_atoms)(z, key, eval)

        v = jnp.expand_dims(v, -2)
        a = jnp.reshape(a, (self.action_dim, n_atoms))
        q_logits = v + a - a.mean(-2, keepdims=True)

        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * self.support, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)

        return NetworkOutputs(q_values=q_values, q_logits=q_logits)


class Rainbow(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        n_steps: int = 3,
        v_max: float = 10.0,
        v_min: Optional[float] = None,
        n_atoms: int = 51,
        optim_kwargs: dict = {"learning_rate": 3e-4},
        init_eps: float = 1.0,
        min_eps: float = 0.01,
        schedule_len: int = 5000,
        use_rmsprop: bool = False,
        seed: Optional[int] = None,
    ):
        seed = seed or np.random.randint(0, int(2e16))
        logging.info(f"Seed: {seed}")
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        self.env = env

        dummy = jnp.zeros(env.observation_space.shape)
        params = critic.init(init_key, dummy, init_key)
        self.targ_params = copy.deepcopy(params)
        self.targ_freq = targ_freq

        self.n_steps = n_steps
        v_min = v_min or -v_max
        support = jnp.linspace(v_min, v_max, n_atoms)

        if use_rmsprop:
            _optimizer = optax.rmsprop(**optim_kwargs)
        else:
            _optimizer = optax.adam(**optim_kwargs)

        optimizer = optax.chain(optax.clip_by_global_norm(10.0), _optimizer)

        self.ts = TrainState.create(apply_fn=critic.apply, params=params, tx=optimizer)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

        def _step(train_state: TrainState, state: np.ndarray, epsilon, key, eval=False):
            eps_key, noise_key = jax.random.split(key)
            q_values = train_state.apply_fn(
                train_state.params, state, noise_key, eval
            ).q_values
            action = distrax.EpsilonGreedy(q_values, epsilon).sample(seed=eps_key)
            return action

        self._step = jax.jit(_step)

        def n_step_returns(gamma, r):
            def _body(acc, xs):
                returns = xs
                acc = returns + gamma * acc
                return acc, acc

            returns, _ = jax.lax.scan(_body, 0.0, r, reverse=True)
            return returns

        _batch_n_step_returns = jax.vmap(n_step_returns, in_axes=(None, 0))

        _batch_categorical_double_q_learning = jax.vmap(
            rlax.categorical_double_q_learning, in_axes=(None, 0, 0, 0, 0, None, 0, 0)
        )

        def _update(train_state: TrainState, targ_params, s, a, r, s_p, d, w, key):
            def loss_fn(params, train_state: TrainState, s, a, r, s_p, d, w, key):
                keys = jax.random.split(key, 3)

                _keys = jax.random.split(keys[0], len(a))
                q_logits = jax.vmap(train_state.apply_fn, (None, 0, 0))(
                    params, s, _keys
                ).q_logits

                _keys = jax.random.split(keys[1], len(a))
                q_p_logits = jax.vmap(train_state.apply_fn, (None, 0, 0))(
                    targ_params, s_p, _keys
                ).q_logits

                _keys = jax.random.split(keys[2], len(a))
                q_p = jax.vmap(train_state.apply_fn, (None, 0, 0))(
                    params, s_p, _keys
                ).q_values

                discount = jnp.power(gamma, self.n_steps) * (1 - d)

                # TODO: try find an example of this function being used and make sure you're using it right re: logits
                error = _batch_categorical_double_q_learning(
                    support, q_logits, a, r, discount, support, q_p_logits, q_p
                )
                mse = jnp.mean(error * w)
                return mse, error

            r = _batch_n_step_returns(0.99, r)
            a = jnp.squeeze(a, -1)
            d = jnp.squeeze(d, -1)
            w = jnp.squeeze(w, -1)

            grads, error = jax.grad(loss_fn, has_aux=True)(
                train_state.params, train_state, s, a, r, s_p, d, w, key
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, error

        self._update = jax.jit(_update)

    def update(self, batches):
        self.key, update_key = jax.random.split(self.key)
        self.ts, error = self._update(
            self.ts,
            self.targ_params,
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
            batches["w"],
            update_key,
        )

        if self.ts.step % self.targ_freq == 0:
            self.targ_params = self.ts.params

        return {"idxs": batches["idxs"], "p": np.asarray(jnp.abs(error))}

    def step(self, state):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.ts, state, self.eps, act_key)
        action = np.asarray(action)
        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action.squeeze(), {}

    def eval_step(self, state: np.ndarray):
        self.key, act_key = jax.random.split(self.key)
        action = self._step(self.ts, state, 0.001, act_key, eval=True)
        action = np.asarray(action)
        return action.squeeze()


def main():
    # env = gym.make("CartPole-v1")
    env = gym.make("LunarLander-v2")

    runner = crl.OffPolicyRunner(
        env=env,
        agent=Rainbow(
            env,
            Q_critic(action_dim=2, support=jnp.linspace(-250, 250, 51)),
            targ_freq=250,
            schedule_len=100_000,
            v_max=250,
            optim_kwargs={"learning_rate": 1e-4},
        ),
        buffer=crl.buffers.PrioritisedBuffer(env, n_steps=3),
        rollout_len=4,
        batch_size=64,
        n_step=3,
    )
    runner.run(rollouts=50_000, eval_freq=5_000)


if __name__ == "__main__":
    main()

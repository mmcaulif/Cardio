"""Data regularised Q from 'Image Augmentation Is All You Need: Regularizing
Deep Reinforcement Learning from Pixels' for discrete environments (Atari
100k).

Paper: https://arxiv.org/abs/2004.13649
Hyperparameters: page 17
Experiment details: page 7
Image augmentation details: page 22

DQN with duelling nets, n-step returns, tuned
hyperpameters and M/K random augmentations applied to S/S_p respectively.

Notes:
Target networks are seemingly removed as target update period = 1.

To do:
* Image augmentation
* Benchmarking (Atari 100k)
* Review differences between DrQ and DrQ(e) from Deep RL at the edge of the
    statistical precipice: Difference is purely in the epsilon parameters,
    differences can be found here:
    https://github.com/google/dopamine/tree/master/dopamine/labs/atari_100k/configs
"""

import functools
import logging
from typing import Optional

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


@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _crop_with_indices(img, x, y, cropped_shape):
    cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
    return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
    """Random crop an image."""
    batch_size, width, height = cropped_shape[:-1]
    key_x, key_y = jax.random.split(key, 2)
    x = jax.random.randint(
        key_x, shape=(batch_size,), minval=0, maxval=img.shape[1] - width
    )
    y = jax.random.randint(
        key_y, shape=(batch_size,), minval=0, maxval=img.shape[2] - height
    )
    return _crop_with_indices(img, x, y, cropped_shape)


def _intensity_aug(key, x, scale=0.05):
    """Follows the code in Schwarzer et al.

    (2020) for intensity augmentation.
    """
    r = jax.random.normal(key, shape=(x.shape[0], 1, 1, 1))
    noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
    return x * noise


def drq_image_augmentation(obs, key, img_pad=4):
    """Padding and cropping for DrQ."""
    flat_obs = obs.reshape(-1, *obs.shape[-3:])
    paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
    cropped_shape = flat_obs.shape
    # The reference uses ReplicationPad2d in pytorch, but it is not available
    # in Jax. Use 'edge' instead.
    flat_obs = jnp.pad(flat_obs, paddings, "edge")
    key1, key2 = jax.random.split(key, num=2)
    cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
    # cropped_obs = _random_crop(key2, flat_obs, cropped_shape)
    aug_obs = _intensity_aug(key1, cropped_obs)
    return aug_obs.reshape(*obs.shape)


class Q_critic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Dense(64)(state))
        z = nn.relu(nn.Dense(64)(z))
        v = nn.Dense(1)(z)
        a = nn.Dense(self.action_dim)(z)
        v = jnp.expand_dims(v, -2)
        q = v + a - a.mean(-2, keepdims=True)
        return q.squeeze()


class DrQ(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        n_steps: int = 10,
        optim_kwargs: dict = {"learning_rate": 1e-4, "eps": 0.00015},
        init_eps: float = 1.0,
        min_eps: float = 0.1,
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

        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(**optim_kwargs),
        )

        self.ts = TrainState.create(apply_fn=critic.apply, params=params, tx=optimizer)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

        def _step(train_state: TrainState, state: np.ndarray, epsilon, key):
            q_values = train_state.apply_fn(train_state.params, state)
            action = distrax.EpsilonGreedy(q_values, epsilon).sample(seed=key)
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

        def _update(train_state: TrainState, s, a, r, s_p, d, key):
            def loss_fn(params, apply_fn, s, a, r, s_p, d, key):
                keys = jax.random.split(key)

                _keys = jax.random.split(keys[0], len(s))
                aug_s = jax.vmap(drq_image_augmentation)(s, _keys)
                q = jax.vmap(apply_fn, in_axes=(None, 0))(params, aug_s)

                _keys = jax.random.split(keys[1], len(s))
                aug_s_p = jax.vmap(drq_image_augmentation)(s_p, _keys)
                q_p = jax.vmap(apply_fn, in_axes=(None, 0))(params, aug_s_p)
                discount = jnp.power(gamma, n_steps) * (1 - d)

                error = jax.vmap(rlax.q_learning)(q, a, r, discount, q_p)
                mse = jnp.mean(jnp.square(error))
                return mse

            r = _batch_n_step_returns(gamma, r)
            a = jnp.squeeze(a, -1)
            d = jnp.squeeze(d, -1)
            grads = jax.grad(loss_fn)(
                train_state.params, train_state.apply_fn, s, a, r, s_p, d, key
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state

        self._update = jax.jit(_update)

    def update(self, batches):
        self.key, update_key = jax.random.split(self.key)
        self.ts = self._update(
            self.ts,
            batches["s"],
            batches["a"],
            batches["r"],
            batches["s_p"],
            batches["d"],
            update_key,
        )
        return {}

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
    env = gym.make("CartPole-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=DrQ(env, Q_critic(2)),
        buffer=crl.buffers.TreeBuffer(env=env, capacity=100_000, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
    )
    runner.run(rollouts=98_400)


if __name__ == "__main__":
    main()

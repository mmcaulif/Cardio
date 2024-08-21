import logging
from typing import Optional

import envpool
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
from cardio_rl.wrappers import EnvPoolWrapper


class Q_critic(nn.Module):
    act_dim: int
    n_atoms: int = 51

    @nn.compact
    def __call__(self, state):
        state /= 255
        z = nn.relu(nn.Conv(32, (8, 8), strides=4)(state))
        z = nn.relu(nn.Conv(64, (4, 4), strides=2)(z))
        z = nn.relu(nn.Conv(64, (3, 3), strides=1)(z))
        z = jnp.reshape(z, (z.shape[0], -1))

        z = nn.relu(nn.Dense(512)(z))
        v = nn.Dense(self.n_atoms)(z)
        a = nn.Dense(self.act_dim * self.n_atoms)(z)

        v = jnp.expand_dims(v, -2)
        a = jnp.reshape(a, (z.shape[0], self.act_dim, self.n_atoms))
        q = v + a - a.mean(-2, keepdims=True)
        return q


class Rainbow(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"learning_rate": 3e-4},
        init_eps: float = 1.0,
        min_eps: float = 0.01,
        schedule_len: int = 5000,
        seed: Optional[int] = None,
    ):
        seed = seed or np.random.randint(0, 2e16)
        logging.info(f"Seed: {seed}")
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        self.env = env

        dummy = jnp.zeros((4, 84, 84))
        params = critic.init(init_key, dummy)
        self.targ_params = critic.init(init_key, dummy)

        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(**optim_kwargs)
        )

        self.ts = TrainState.create(apply_fn=critic.apply, params=params, tx=optimizer)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

        atoms = jnp.expand_dims(jnp.linspace(-10, 10, 51), 0)

        def _step(train_state: TrainState, state: np.ndarray):
            q_logits = train_state.apply_fn(train_state.params, state)
            dist = nn.softmax(q_logits, axis=-1)
            q = (dist * atoms).sum(-1)
            action = jnp.argmax(q, axis=-1)
            return action

        self._step = jax.jit(_step)

        def _update(train_state: TrainState, targ_params, s, a, r, s_p, d, w):
            def loss_fn(params, train_state: TrainState, s, a, r, s_p, d, w):
                q_logits = train_state.apply_fn(params, s)
                q_p_logits = train_state.apply_fn(targ_params, s_p)
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
        if np.random.rand() > self.eps:
            action = self._step(self.ts, state)
            action = np.asarray(action).squeeze(0)
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
    env = envpool.make_gymnasium(
        "Freeway-v5", num_envs=1, episodic_life=True, reward_clip=True
    )
    env = EnvPoolWrapper(env)

    eval_env = envpool.make_gymnasium("Freeway-v5", num_envs=1)
    eval_env = EnvPoolWrapper(eval_env)

    runner = crl.OffPolicyRunner(
        env=env,
        agent=Rainbow(
            env=env,
            critic=Q_critic(act_dim=env.action_space.n),
            targ_freq=2_000,
            optim_kwargs={"learning_rate": 0.0001, "eps": 0.00015},
            schedule_len=2_000,
        ),
        buffer=crl.buffers.PrioritisedBuffer(env=env, capacity=100_000, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
        eval_env=eval_env,
    )

    runner.run(rollouts=98_400, eval_freq=10_000, eval_episodes=10)


if __name__ == "__main__":
    main()

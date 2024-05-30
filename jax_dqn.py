# type: ignore

from typing import Any
import distrax
import jax
import numpy as np
import cardio_rl as crl
import gymnasium as gym
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import rlax

from flax.training.train_state import TrainState
from gymnasium import Env

"""
Efficient Rainbow impl checklist:
* [x] Duelling dqn
* [x] Double dqn
* [ ] PER
* [ ] C51
* [x] n-step returns
"""


class DdqnTrainState(TrainState):
    target_params: flax.core.FrozenDict[str, Any]


class DuellingQ(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, s):
        z = nn.relu(nn.Dense(64)(s))
        z = nn.relu(nn.Dense(64)(z))
        a = nn.Dense(self.action_dim)(z)
        v = nn.Dense(1)(z)
        q = v + (a - jnp.mean(a))
        return q


class DDQN(crl.Agent):
    def __init__(self, seed: int = 0):
        self.key = jax.random.PRNGKey(seed)
        self.eps = 0.9
        self.min_eps = 0.05
        schedule_steps = 20_000
        self.ann_coeff = self.min_eps ** (1 / schedule_steps)

    def setup(self, env: Env):
        self.env = env
        self.key, init_key = jax.random.split(self.key)
        model = DuellingQ(action_dim=env.action_space.n)
        params = model.init(init_key, jnp.zeros(env.observation_space.shape[0]))
        optim = optax.adam(1e-4)
        opt_state = optim.init(params)

        self.train_state = DdqnTrainState(
            step=0,
            apply_fn=jax.jit(model.apply),
            params=params,
            opt_state=opt_state,
            tx=optim,
            target_params=params,
        )

    def update(self, batches: list):
        """
        Jax can only scan through an array, need to figure out a way to implement batches of data
        """
        n = jnp.arange(len(batches))
        batches = iter(batches)

        def scan_batch(train_state: DdqnTrainState, n: int):
            data = next(batches)
            @jax.value_and_grad
            def _loss(params, train_state: DdqnTrainState, data: dict):
                q = train_state.apply_fn(params, data["s"])
                q_p_value = train_state.apply_fn(train_state.target_params, data["s_p"])
                q_p_select = train_state.apply_fn(
                    train_state.target_params, data["s_p"]
                )
                discount = (1 - data["d"]) * 0.99
                error = jax.vmap(rlax.double_q_learning)(
                    q,
                    jnp.squeeze(data["a"]),
                    jnp.squeeze(data["r"]),
                    jnp.squeeze(discount),
                    q_p_value,
                    q_p_select,
                )
                loss = jnp.mean(error**2)
                return loss

            loss, grads = _loss(train_state.params, train_state, data)
            train_state = train_state.apply_gradients(grads=grads)

            target_params = optax.incremental_update(
                train_state.params, train_state.target_params, 0.005
            )

            train_state = train_state.replace(target_params=target_params)
            return train_state, loss
        
        self.train_state, _ = jax.lax.scan(scan_batch, self.train_state, n)

    def step(self, state):
        # @jax.jit
        def _step(key, s: np.array, train_state: DdqnTrainState, eps: float):
            key, act_key = jax.random.split(key)
            q = train_state.apply_fn(train_state.target_params, s)
            a = distrax.EpsilonGreedy(q, eps).sample(seed=act_key)
            return a, key

        action, self.key = _step(self.key, state, self.train_state, self.eps)
        action = np.array(action)
        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}


def main():
    env_name = "CartPole-v1"  # CartPole-v1 LunarLander-v2

    runner = crl.OffPolicyRunner(
        env=gym.make(env_name),
        agent=DDQN(),
        rollout_len=4,
        batch_size=32,
        warmup_len=10_000,
        n_batches=1,
    )
    # runner.evaluate(episodes=500)
    runner.run(10_000)


if __name__ == "__main__":
    main()

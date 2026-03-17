import copy
import time
from functools import partial

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
import rlax
from flax.training.train_state import TrainState

# ------------------------
# Simulated workload
# ------------------------

NUM_ITERS = 2_500
STEPS_PER_ITER = 32
ENV_NAME = 'MinAtar/SpaceInvaders-v1'


class Q_critic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        z = nn.relu(nn.Conv(16, (3, 3), strides=1)(state))
        z = jnp.reshape(z, -1)
        z = nn.relu(nn.Dense(128)(z))
        q = nn.Dense(self.act_dim)(z)
        return q


class Gatherer:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state, _ = self.env.reset()

        self.key = jax.random.PRNGKey(0)
        self.key, init_key = jax.random.split(self.key)

        env = gym.make(env_name)
        dummy = env.observation_space.sample()
        critic = Q_critic(act_dim=env.action_space.n)
        params = critic.init(init_key, dummy)
        self.q = jax.jit(lambda s: critic.apply(params, s))

    def collect(self, num_steps):
        """Collect (s, a, r, s', done) tuples."""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for _ in range(num_steps):
            q_values = self.q(self.state)
            self.key, act_key = jax.random.split(self.key)
            action = distrax.EpsilonGreedy(q_values, 0.01).sample(seed=act_key)
            action = np.asarray(action).squeeze()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            states.append(self.state.astype(np.float32))
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.astype(np.float32))
            dones.append(done)

            self.state = next_state
            if done:
                self.state, _ = self.env.reset()
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32)
        )


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

    grads, _ = jax.grad(loss_fn, has_aux=True)(
        ts.params, ts.apply_fn, s, a, r, s_p, d
    )
    new_ts = ts.apply_gradients(grads=grads)
    return new_ts


class Agent:
    def __init__(
        self,
        env_name: str,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"learning_rate": 1e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
        use_rmsprop: bool = False,
    ):
        self.key = jax.random.PRNGKey(0)
        self.key, init_key = jax.random.split(self.key)

        env = gym.make(env_name)
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

        self._update = jax.jit(partial(_update, gamma=gamma))

    def update(self, batch):
        s, a, r, s_p, d = batch
        self.ts = self._update(
            self.ts,
            self.targ_params,
            s,
            a,
            r,
            s_p,
            d,
        )

        if self.ts.step % self.targ_freq == 0:
            self.targ_params = self.ts.params
    

def sync_loop():
    """Synchronous collection + training."""
    start = time.time()
    gatherer = Gatherer(env_name=ENV_NAME)
    agent = Agent(env_name=ENV_NAME)

    start = time.time()

    for _ in range(NUM_ITERS):
        batch = gatherer.collect(STEPS_PER_ITER)
        agent.update(batch)

    return time.time() - start


def async_loop():
    """Overlapping collection and training using Ray."""
    gatherer = ray.remote(Gatherer).remote(env_name=ENV_NAME)
    agent = ray.remote(Agent).remote(env_name=ENV_NAME)

    start = time.time()

    # Start first collection
    collection_future = gatherer.collect.remote(STEPS_PER_ITER)

    for _ in range(NUM_ITERS):
        # Wait for collection
        batch = ray.get(collection_future)

        # Start training in background
        train_future = agent.update.remote(batch)

        # Start collecting next batch immediately
        collection_future = gatherer.collect.remote(STEPS_PER_ITER)

        # Wait for training to complete
        ray.get(train_future)

    return time.time() - start


if __name__ == "__main__":
    ray.init()
    sync_time = sync_loop()
    async_time = async_loop()
    print(f"Synchronous total time: {sync_time:.2f} sec")
    print(f"Async (overlapped) total time: {async_time:.2f} sec")
    print(f"Speedup: {sync_time / async_time:.2f}x")

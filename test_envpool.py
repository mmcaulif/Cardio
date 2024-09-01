import time

import envpool
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange


class Network(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, state):
        state /= 255
        z = nn.relu(nn.Conv(32, (8, 8), strides=4)(state))
        z = nn.relu(nn.Conv(64, (4, 4), strides=2)(z))
        z = nn.relu(nn.Conv(64, (3, 3), strides=1)(z))
        z = jnp.reshape(z, (z.shape[0], -1))
        z = nn.relu(nn.Dense(512)(z))
        q = nn.Dense(self.act_dim)(z)
        return q


### Envpool

env: gym.Env = envpool.make(
    "Pong-v5",
    env_type="gymnasium",
    num_envs=1,
    episodic_life=True,
    reward_clip=True,
    repeat_action_probability=0.25,
    full_action_space=False,
)

print(env.observation_space.shape)
print(env.action_space.n)

s, _ = env.reset()
model = Network(env.action_space.n)

params = model.init(jax.random.PRNGKey(0), (1, 4, 84, 84))
apply_fn = jax.jit(model.apply)
q = apply_fn(params, s)
print(q.shape)

t = time.time()
steps = 100_000
episodes = 0

for _ in trange(steps):
    a = apply_fn(params, s).argmax(-1)
    a = np.asarray(a)
    s, r, term, trun, _ = env.step(a)
    if term or trun:
        s, _ = env.reset()
        episodes += 1

print(f"Envpool took {time.time() - t}s for {episodes} episodes and {steps} env steps")

exit()

### MinAtar

time.sleep(5)

env = gym.make("MinAtar/Freeway-v1")

t = time.time()
steps = 100_000
episodes = 0

s, _ = env.reset()
for _ in trange(steps):
    a = env.action_space.sample()
    s, r, term, trun, _ = env.step(a)
    if term or trun:
        s, _ = env.reset()
        episodes += 1

print(f"MinAtar took {time.time() - t}s for {episodes} episodes and {steps} env steps")

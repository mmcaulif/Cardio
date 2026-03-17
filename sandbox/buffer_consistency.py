
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

import cardio_rl as crl
from cardio_rl.buffers.eff_buffer import EffBuffer
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.toy_env import ToyEnv

np.random.seed(42)

# env = ToyEnv()
env = gymnasium.make("CartPole-v1")
buffer_1 = TreeBuffer(env, capacity=1000, batch_size=1)
buffer_2 = EffBuffer(env, capacity=1000, batch_size=1)
s, _ = env.reset()

for i in range(200):
    a = env.action_space.sample()
    s_p, r, d, t, _ = env.step(a)
    transition = {
        "s": s,
        "a": a,
        "r": r,
        "s_p": s_p,
        "d": d,
    }
    expanded_transition = crl.tree.stack([transition])
    buffer_1.store(expanded_transition, 1)
    buffer_2.store(expanded_transition, 1)

    # print(f"Buffer 1 states: {buffer_1.get('s')}")
    # print(f"Buffer 2 states: {buffer_2.get('s')}")

    if i > 10:
        sample2 = buffer_2.sample()
        sample1 = buffer_1.sample(sample_indxs=sample2['idxs'])

        if not np.allclose(sample2['s_p'], sample1['s_p']):
            print(f"Sample 1: {sample1}")
            print(f"Sample 2: {sample2}")
            print('\n')

    s = s_p
    if d or t:
        s, _ = env.reset()
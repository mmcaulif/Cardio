"""Module for checking buffer memory usage in a Gymnasium Atari environment."""


# import cardio_rl as crl
# from cardio_rl.toy_env import ToyEnv
# from cardio_rl.buffers.eff_mixed_buffer import EffMixedBuffer


# env = ToyEnv()
# runner = crl.Runner.off_policy(
#     env=env,
#     agent=crl.Agent(env),
#     rollout_len=4,
#     warmup_len=0,
#     buffer=EffMixedBuffer(env),
# )

# data = runner.step()
# recent_index = data["idxs"][0]
# table_pos = runner.buffer.pos - 1 % runner.buffer.capacity
# print(recent_index, table_pos)
# print(data)

import gymnasium as gym

import cardio_rl as crl
from cardio_rl.wrappers import AtariWrapper


# Create a Gymnasium environment
env = gym.make("BreakoutNoFrameskip-v4")
env = AtariWrapper(env)

buffer_1 = crl.buffers.MixedBuffer(
    env=env,
    batch_size=32,
)

print(f"Buffer 1 bytes footprint: {buffer_1.nbytes / 1e9}")
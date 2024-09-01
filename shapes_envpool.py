import envpool
import gymnasium as gym
import numpy as np

from cardio_rl.wrappers import EnvPoolWrapper

### Envpool

env_1: gym.Env = envpool.make(
    "Pong-v5",
    env_type="gymnasium",
    num_envs=1,
    episodic_life=True,
    reward_clip=True,
    repeat_action_probability=0.25,
    full_action_space=False,
)

print(env_1.observation_space.shape)
print(env_1.action_space.n)
s, _ = env_1.reset()
print(s.shape)
action = np.expand_dims(env_1.action_space.sample(), 0).astype(np.int32)
print(action, action.shape, action.dtype)
s, r, d, t, _ = env_1.step(action)
print(s.shape, r.shape, d.shape, t.shape)

print("\nWrapped env:")
env_2 = EnvPoolWrapper(env=env_1)
print(env_2.observation_space.shape)
print(env_2.action_space.n)
s, _ = env_2.reset()
print(s.shape)
action = env_2.action_space.sample()
print(action, action.shape, action.dtype)
s, r, d, t, _ = env_2.step(action)
print(s.shape, r.shape, d.shape, t.shape)

print("\nMinAtar:")
print("\nWrapped env:")
ma_env = gym.make("MinAtar/Freeway-v1")
print(ma_env.observation_space.shape)
print(ma_env.action_space.n)
s, _ = ma_env.reset()
print(s.shape)
action = ma_env.action_space.sample()
print(action, action.shape, action.dtype)
s, r, d, t, _ = ma_env.step(action)
print(s.shape)

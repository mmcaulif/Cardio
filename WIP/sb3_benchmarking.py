import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise

# env = gym.make("CartPole-v1")
# print('test')
# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=50_000)

# env = gym.make('MountainCarContinuous-v0')
# print('test')
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
# model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise)
# model.learn(total_timesteps=300_000, log_interval=10_000)

env_name = "BipedalWalker-v3"
env = gym.make(env_name)
print("test")
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    action_noise=action_noise,
    gamma=0.98,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    buffer_size=300_000,
    learning_starts=10_000,
    learning_rate=7.3e-4,
    policy_kwargs=dict(net_arch=[400, 300]),
    # stats_window_size=40,
    tensorboard_log="BipedalWalker-v3/",
)

model.learn(
    total_timesteps=300_000, log_interval=6, tb_log_name="vanilla", progress_bar=True
)

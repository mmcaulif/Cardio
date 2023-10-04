import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import RescaleAction


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

"""
SB3 zoo hyperparams:
  n_timesteps: !!float 3e5
  policy: 'MlpPolicy'
  gamma: 0.98
  buffer_size: 200000
  learning_starts: 10000
  noise_type: 'normal'
  noise_std: 0.1
  gradient_steps: -1
  train_freq: [1, "episode"]
  learning_rate: !!float 1e-3
  policy_kwargs: "dict(net_arch=[400, 300])"
"""

model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    action_noise=action_noise, 
    gamma=0.98,
    train_freq=1,
    buffer_size=200_000,
    learning_starts=10_000,
    learning_rate=1e-3,
    policy_kwargs=dict(net_arch=[400, 300]),
    stats_window_size=50,
    tensorboard_log='implicit_event_tables/tb_logs/')

model.learn(
    total_timesteps=160_000,
    log_interval=10,
    tb_log_name='sb3_td3_benchmark',
    progress_bar=True
    )
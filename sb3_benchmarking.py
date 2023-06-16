import gymnasium as gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")

print('test')
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

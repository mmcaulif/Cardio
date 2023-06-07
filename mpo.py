
import gymnasium as gym
from src import Runner, get_offpolicy_runner
from src.policies import Base_policy

env = gym.make('CartPole-v1')
runner = get_offpolicy_runner(
	env, 
	Base_policy(env),
	length=256,
	capacity=100000, 
	batch_size=64, 
	train_after=1000)

print(runner.get_batch())
import gymnasium as gym
from runner import Runner

env = gym.make('CartPole-v1')

runner = Runner(env, 32, True, True, 100000, 1, 0)

for t in range(20):
    print(len(runner.buffer))

    if t == 10:
        runner.flush_buffer()

    batch = runner.get_batch(None, 'random')
    print(len(batch))
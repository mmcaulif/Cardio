"""
Data Efficient Rainbow from 'When to use parametric models in
reinforcement learning?' for discrete environments (Atari 100k).

Paper:
Hyperparameters: https://github.com/google/dopamine/blob/master/dopamine/labs/atari_100k/configs/DER.gin
Experiment details:

Rainbow with tuned hyperparameters for sample efficiency

Notes:

To do:
* C51
* Benchmarking (Atari 100k)
"""

import gymnasium as gym
from rainbow import Rainbow

import cardio_rl as crl


def main():
    env = gym.make("CartPole-v1")
    runner = crl.OffPolicyRunner(
        env,
        agent=Rainbow(env),
        buffer=crl.buffers.PrioritisedBuffer(env, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
    )

    runner.run(98_400)


if __name__ == "__main__":
    main()

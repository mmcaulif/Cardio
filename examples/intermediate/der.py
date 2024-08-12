import gymnasium as gym
from rainbow import Rainbow

import cardio_rl as crl


def main():
    env = gym.make("CartPole-v1")
    agent = Rainbow(env)
    runner = crl.OffPolicyRunner(
        env,
        agent,
        buffer=crl.buffers.PrioritisedBuffer(env, n_steps=10),
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
    )

    runner.run(98_400)


if __name__ == "__main__":
    main()

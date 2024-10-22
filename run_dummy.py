import cardio_rl as crl


def main():
    env = crl.toy_env.ToyEnv()
    runner = crl.OffPolicyRunner(
        env=env, agent=crl.Agent(env), rollout_len=4, batch_size=32, warmup_len=0
    )
    runner.run(rollouts=50_000, eval_freq=1_250)


if __name__ == "__main__":
    main()

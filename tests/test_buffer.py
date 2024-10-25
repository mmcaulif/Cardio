import cardio_rl as crl
from cardio_rl.buffers.tree_buffer import TreeBuffer
from cardio_rl.toy_env import ToyEnv


class TestTreeBuffer:
    def test_init_shape(self):
        capacity = 10_000
        env = ToyEnv()
        buffer = TreeBuffer(env, capacity)
        assert buffer.table["s"].shape == (capacity, 5)
        assert buffer.table["a"].shape == (capacity, 1)
        assert buffer.table["r"].shape == (capacity, 1)
        assert buffer.table["s_p"].shape == (capacity, 5)
        assert buffer.table["d"].shape == (capacity, 1)

    def test_buffer_with_runner(self):
        env = crl.toy_env.ToyEnv()
        runner = crl.Runner.off_policy(
            env=env,
            agent=crl.Agent(env),
            buffer_kwargs={"batch_size": 32},
            rollout_len=4,
        )
        sample = runner.step()
        assert sample["s"].shape == (32, 5)
        assert sample["a"].shape == (32, 1)
        assert sample["r"].shape == (32, 1)
        assert sample["s_p"].shape == (32, 5)
        assert sample["d"].shape == (32, 1)

    def test_extra_specs(self):
        env = crl.toy_env.ToyEnv()
        runner = crl.Runner.off_policy(
            env=env,
            agent=crl.Agent(env),
            buffer_kwargs={"batch_size": 32},
            rollout_len=4,
            warmup_len=0,
            extra_specs={"example": [1]},
        )
        assert runner.buffer.table["example"].shape == (runner.buffer.capacity, 1)

    def test_sample(self):
        steps = 1

        env = ToyEnv()
        buffer = TreeBuffer(env, batch_size=steps)
        s, _ = env.reset()

        for _ in range(steps):
            a = env.action_space.sample()
            s_p, r, d, t, _ = env.step(a)
            transition = {
                "s": s,
                "a": a,
                "r": r,
                "s_p": s_p,
                "d": d,
            }
            expanded_transition = crl.tree.stack([transition])
            buffer.store(expanded_transition, 1)
            s = s_p
            if d or t:
                s, _ = env.reset()

        sample = buffer.sample()
        assert sample["s"].shape == (steps, 5)
        assert sample["a"].shape == (steps, 1)
        assert sample["r"].shape == (steps, 1)
        assert sample["s_p"].shape == (steps, 5)
        assert sample["d"].shape == (steps, 1)

    def test_trajectories(self):
        batch_size = 4
        trajectory = 3
        steps = 20

        env = ToyEnv()
        buffer = TreeBuffer(env, batch_size=batch_size, trajectory=trajectory)
        s, _ = env.reset()

        for _ in range(steps):
            a = env.action_space.sample()
            s_p, r, d, t, _ = env.step(a)
            transition = {
                "s": s,
                "a": a,
                "r": r,
                "s_p": s_p,
                "d": d,
            }
            expanded_transition = crl.tree.stack([transition])
            buffer.store(expanded_transition, 1)
            s = s_p
            if d or t:
                s, _ = env.reset()

        sample = buffer.sample()
        assert sample["s"].shape == (4, 3, 5)
        assert sample["a"].shape == (4, 3, 1)
        assert sample["r"].shape == (4, 3, 1)
        assert sample["s_p"].shape == (4, 3, 5)
        assert sample["d"].shape == (4, 3, 1)

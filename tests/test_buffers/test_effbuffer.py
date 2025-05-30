import pytest

import cardio_rl as crl
from cardio_rl.buffers.eff_buffer import EffBuffer
from cardio_rl.toy_env import ToyEnv


class TestEffBuffer:
    @pytest.mark.parametrize("capacity", [(10_000), (100_000), (10), (123456)])
    def test_init_shape(self, capacity):
        env = ToyEnv()
        buffer = EffBuffer(env, capacity)
        assert buffer.table["s"].shape == (capacity, 5)
        assert buffer.table["a"].shape == (capacity, 1)
        assert buffer.table["r"].shape == (capacity, 1)
        assert buffer.table["d"].shape == (capacity, 1)

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

    @pytest.mark.parametrize("steps", [(4), (32), (16)])
    def test_batchsize(self, steps):
        env = ToyEnv()
        buffer = EffBuffer(env, batch_size=steps)
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

    @pytest.mark.parametrize("batch_size", [(32), (2), (1024), (1), (50_000)])
    def test_batchsize_with_runner(self, batch_size):
        env = crl.toy_env.ToyEnv()
        runner = crl.Runner.off_policy(
            env=env,
            agent=crl.Agent(env),
            buffer_kwargs={"batch_size": batch_size},
            warmup_len=10_000,
        )
        sample = runner.step()
        k = min(batch_size, len(runner.buffer))
        assert sample["s"].shape == (k, 5)
        assert sample["a"].shape == (k, 1)
        assert sample["r"].shape == (k, 1)
        assert sample["s_p"].shape == (k, 5)
        assert sample["d"].shape == (k, 1)

    @pytest.mark.parametrize("n_batches", [(1), (2), (16)])
    def test_n_batches(self, n_batches):
        env = ToyEnv()
        buffer = EffBuffer(env, n_batches=n_batches)
        s, _ = env.reset()

        for _ in range(100):
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
        if n_batches == 1:
            assert len(sample["r"]) == buffer.batch_size
        else:
            assert len(sample) == n_batches
            assert len(sample[0]["r"]) == buffer.batch_size

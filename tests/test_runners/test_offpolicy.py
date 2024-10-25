import pytest

import cardio_rl as crl


class TestOffpolicy:
    @pytest.mark.parametrize(
        "capacity, batch_size, n_steps, trajectory, n_batches",
        [(1_000, 4, 2, 1, 1), (100_000, 64, 1, 5, 3)],
    )
    def test_buffer_init(self, capacity, batch_size, n_steps, trajectory, n_batches):
        env = crl.toy_env.ToyEnv()
        runner = crl.Runner.off_policy(
            env=env,
            agent=crl.Agent(env),
            buffer_kwargs={
                "capacity": capacity,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "trajectory": trajectory,
                "n_batches": n_batches,
            },
            warmup_len=0,
        )
        assert runner.buffer.capacity == capacity
        assert runner.buffer.batch_size == batch_size
        assert runner.buffer.n_steps == n_steps
        assert runner.buffer.trajectory == trajectory
        assert runner.buffer.n_batches == n_batches

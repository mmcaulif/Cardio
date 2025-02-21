import pytest

import cardio_rl as crl
import gymnasium as gym
from cardio_rl.buffers.mixed_buffer import MixedBuffer
from cardio_rl.toy_env import ToyEnv


class TestTreeBuffer:
    @pytest.mark.parametrize(
        "warmup, rollout_len, capacity",
        [
            (64, 2, 1_000),
            (1_000, 16, 10_000),
            (8, 8, 10),
        ],
    )
    def test_init_shape(self, warmup, rollout_len, capacity):
        env = ToyEnv()
        runner = crl.Runner.off_policy(
            env=env,
            agent=crl.Agent(env),
            rollout_len=rollout_len,
            warmup_len=warmup,
            buffer=MixedBuffer(env, capacity),
        )
        data = runner.step()
        recent_index = data["idxs"][0]
        table_pos = runner.buffer.pos - 1 % capacity
        assert recent_index == table_pos

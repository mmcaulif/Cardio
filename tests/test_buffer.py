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

    def test_sample(self):
        env = ToyEnv()
        buffer = TreeBuffer(env)
        s, _ = env.reset()

        steps = 1

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

        sample = buffer.sample(steps)
        assert sample["s"].shape == (steps, 5)
        assert sample["a"].shape == (steps, 1)
        assert sample["r"].shape == (steps, 1)
        assert sample["s_p"].shape == (steps, 5)
        assert sample["d"].shape == (steps, 1)

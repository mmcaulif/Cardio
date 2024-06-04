import cardio_rl as crl
from cardio_rl.toy_env import ToyEnv
from cardio_rl.buffers import TreeBuffer

class TestTreeBuffer:
    def test_init(self):
        cap = 10_000
        env = ToyEnv()
        buffer = TreeBuffer(env, capacity=cap)
        assert buffer.table['s'].shape == (cap, 5)
        assert buffer.table['a'].shape == (cap, 1)
        assert buffer.table['r'].shape == (cap, 1)
        assert buffer.table['s_p'].shape == (cap, 5)
        assert buffer.table['d'].shape == (cap, 1)

    def test_sample(self):
        env = ToyEnv()
        buffer = TreeBuffer(env)
        s, _ = env.reset()
        a = env.action_space.sample()
        s_p, r, d, _, _ = env.step(a)
        transition = {
            "s": s,
            "a": a,
            "r": r,
            "s_p": s_p,
            "d": d,
        }
        expanded_transition = crl.tree.stack([transition])
        buffer.store(expanded_transition)
        sample = buffer.sample(1)
        
        assert sample['s'].shape == (1, 5)
        assert sample['a'].shape == (1, 1)
        assert sample['a'] == a
        assert sample['r'].shape == (1, 1)
        assert sample['s_p'].shape == (1, 5)
        assert sample['d'] == d

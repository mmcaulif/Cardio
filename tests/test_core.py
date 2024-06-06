import cardio_rl as crl
import gymnasium as gym
import numpy as np
from cardio_rl.toy_env import ToyEnv


class TestStack:
    """
    Add reparameterisation for multiple different shapes/data
    """

    def test_data(self):
        d1 = {
            "a": np.array([1.0]),
            "b": np.array([0.1]),
        }
        d2 = {
            "a": np.array([2.0]),
            "b": np.array([0.2]),
        }
        stacked_dicts = crl.tree.stack([d1, d2])
        cond_a = np.allclose(stacked_dicts["a"], np.array([[1.0], [2.0]]))
        cond_b = np.allclose(stacked_dicts["b"], np.array([[0.1], [0.2]]))
        assert cond_a and cond_b

    def test_shape(self):
        d1 = {
            "a": np.ones([100]),
            "b": np.ones([10, 3]),
        }
        d2 = {
            "a": np.ones([100]),
            "b": np.ones([10, 3]),
        }
        stacked_dicts = crl.tree.stack([d1, d2])
        cond_a = stacked_dicts["a"].shape == (2, 100)
        cond_b = stacked_dicts["b"].shape == (2, 10, 3)
        assert cond_a and cond_b

    def test_added_dim(self):
        d = {
            "a": np.ones([100]),
        }
        stacked_dicts = crl.tree.stack([d])
        assert stacked_dicts["a"].shape == (1, 100)


class TestConcatenate:
    """
    Add reparameterisation for multiple different shapes/data
    """

    def test_data(self):
        d1 = {
            "a": np.array([1.0]),
            "b": np.array([0.1]),
        }
        d2 = {
            "a": np.array([2.0]),
            "b": np.array([0.2]),
        }
        stacked_dicts = crl.tree.concatenate([d1, d2])
        cond_a = np.allclose(stacked_dicts["a"], np.array([1.0, 2.0]))
        cond_b = np.allclose(stacked_dicts["b"], np.array([0.1, 0.2]))
        assert cond_a and cond_b

    def test_shape(self):
        d1 = {
            "a": np.ones([100]),
            "b": np.ones([10, 3]),
        }
        d2 = {
            "a": np.ones([100]),
            "b": np.ones([10, 3]),
        }
        stacked_dicts = crl.tree.concatenate([d1, d2])
        cond_a = stacked_dicts["a"].shape == (200,)
        cond_b = stacked_dicts["b"].shape == (20, 3)
        assert cond_a and cond_b


class TestAgent:
    def test_init(self):
        env = ToyEnv()
        agent = crl.Agent(env)
        assert isinstance(agent.env, ToyEnv)

    def test_step(self):
        env = ToyEnv()
        agent = crl.Agent(env)
        s_t, info = env.reset()
        a_t, extras = agent.step(s_t)
        del info
        del a_t
        assert isinstance(extras, dict)

    def test_eval_step(self):
        env = ToyEnv()
        agent = crl.Agent(env)
        s_t, info = env.reset()
        a_t, extras = agent.eval_step(s_t)
        del info
        del a_t
        assert isinstance(extras, dict)


class TestEnv:
    def test_length(self):
        length = 20
        env = ToyEnv(maxlen=length)
        for i in range(length):
            _, _, done, _, _ = env.step(env.action_space.sample())
            if i == length - 1:
                assert done
            else:
                assert not done

    def test_action_sample(self):
        env = ToyEnv()
        for _ in range(100):
            assert env.action_space.sample() in [0, 1]

    def test_action_n(self):
        env = ToyEnv()
        assert env.action_space.n == 2

    def test_action_space(self):
        env = ToyEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_obs_space(self):
        env = ToyEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_action_oracle(self):
        env = ToyEnv()
        done = False
        while not done:
            a_t = env.action_space.sample()
            s_tp1, _, done, _, _ = env.step(a_t)
            assert s_tp1[-1] == a_t

    def test_obs_count(self):
        env = ToyEnv()
        done = False
        i = 1
        while not done:
            s_tp1, _, done, _, _ = env.step(env.action_space.sample())
            assert s_tp1[0] == i
            i += 1


# class TestUtils

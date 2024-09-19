import cardio_rl as crl
from cardio_rl.toy_env import ToyEnv


class TestGatherer:
    def test_step(self):
        env = ToyEnv()
        gatherer = crl.Gatherer()
        gatherer.init_env(env)
        agent = crl.Agent(env)
        rollout_batch = gatherer.step(agent, 1)
        assert len(rollout_batch) == 1

    def test_episode_rollout(self):
        env = ToyEnv()
        gatherer = crl.Gatherer(n_step=3)
        gatherer.init_env(env)
        agent = crl.Agent(env)
        rollout_batch = gatherer.step(agent, -1)
        assert len(rollout_batch) == env.maxlen

    def test_empty_nstep(self):
        env = ToyEnv()
        gatherer = crl.Gatherer(n_step=3)
        gatherer.init_env(env)
        agent = crl.Agent(env)
        rollout_batch = gatherer.step(agent, 2)
        assert len(rollout_batch) == 0

    def test_nstep(self):
        env = ToyEnv()
        gatherer = crl.Gatherer(n_step=3)
        gatherer.init_env(env)
        agent = crl.Agent(env)
        _ = gatherer.step(agent, 2)
        assert len(gatherer.step_buffer) == 2

from collections import deque
from cardio_rl import Gatherer, Runner
from cardio_rl.policies import BasePolicy
from minari import DataCollectorV0
import minari
import gymnasium as gym
import d4rl


class OfflineCollector(Gatherer):
    def __init__(self, env, capacity) -> None:
        super().__init__(env, capacity)
        self.env = env
        self.warmup()

    def warmup(self, net=None, policy=None, length=0, n_step=1):
        if isinstance(self.env, gym.Env):
            return self._warmup_gym(net, policy, length, n_step)

        self._warumup_d4rl()

        return

    def _warumup_d4rl(self):
        dataset = d4rl.qlearning_dataset(self.env)
        print(dataset)

    def _warmup_gym(self, net=None, policy=None, length=1000000, n_step=1):
        if policy == None:
            policy = BasePolicy(self.env)

        self.env = DataCollectorV0(
            self.env, record_infos=False, max_buffer_steps=length
        )

        self.net = net
        gather_buffer = deque()
        step_buffer = deque(maxlen=n_step)

        if length < 1:
            print("cant have n-step less than 1 when using offline gatherer")

        for _ in range(length):
            self.total_steps += 1
            a = policy(self.state, self.net)
            s_p, r, d, t, info = self.env.step(a)

            step_buffer.append([self.state, a, r, s_p, d])
            if len(step_buffer) == n_step:
                if n_step == 1:
                    gather_buffer.append(*list(step_buffer))
                else:
                    gather_buffer.append(list(step_buffer))

            self.state = s_p
            if d or t:
                self.episodes += 1
                self.epsiode_window.append(self.ep_rew)
                self.state, _ = self.env.reset()
                step_buffer = deque(maxlen=n_step)

        self._set_up_loader(data=list(gather_buffer))

        return list(gather_buffer)

    def _set_up_loader(self, data):
        print(len(data))
        dataset = minari.create_dataset_from_collector_env(
            dataset_id="CartPole-v1-test-v0",
            collector_env=self.env,
            author="Manus",
            author_email="mmcaulif@tcd.ie",
        )


def get_offline_runner(env, capacity):
    return Runner(
        env,
        BasePolicy(env),
        length=1,
        sampler=False,
        capacity=0,
        batch_size=256,
        n_step=1,
        collector=OfflineCollector,
        train_after=capacity,
    )


def main():
    env = gym.make("maze2d-umaze-v1")
    runner = get_offline_runner(env, 200)


print("here")
main()

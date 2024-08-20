import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import cardio_rl as crl
from examples.simple.stub_dqn import DQN


class Q_critic(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Q_critic, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, x: th.Tensor):
        x = x.float()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = th.permute(x, (0, 3, 1, 2))
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.reshape(x.size(0), -1)))
        return self.output(x).squeeze(-1)


def main():
    def how_many_rollouts(env_steps: int, warmup_len: int, rollout_len: int) -> int:
        return int((env_steps - warmup_len) / rollout_len)

    env = gym.make("MinAtar/Freeway-v1")  # Matches paper performance
    # env = gym.make("MinAtar/Breakout-v1")

    agent = DQN(
        env=env,
        critic=Q_critic(env.observation_space.shape[-1], env.action_space.n),
        optim_kwargs={
            "lr": 0.00025,
            "alpha": 0.95,
            "centered": True,
            "eps": 0.01,
        },
        init_eps=1.0,
        min_eps=0.1,
        schedule_len=100_000,
        use_rmsprop=True,
    )

    runner = crl.OffPolicyRunner(
        env=env,
        agent=agent,
        capacity=100_000,
        rollout_len=1,
        batch_size=32,
        warmup_len=5_000,
    )

    n_steps = how_many_rollouts(1_000_000, runner.warmup_len, runner.rollout_len)

    runner.run(rollouts=n_steps, eval_freq=25_000, eval_episodes=20)


if __name__ == "__main__":
    main()

import copy

import gymnasium as gym
import jax
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import cardio_rl as crl


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
        if len(x.shape) != 4:
            x.unsqueeze_(0)
        x = th.permute(x, (0, 3, 1, 2))
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.reshape(x.size(0), -1)))
        return self.output(x).squeeze(-1)


class DQN(crl.Agent):
    """All parameters from MinAtar repo:
    https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
    """

    def __init__(self, env: gym.Env):
        self.env = env
        self.critic = Q_critic(env.observation_space.shape[-1], env.action_space.n)
        self.targ_critic = copy.deepcopy(self.critic)
        self.update_count = 0
        self.optimizer = th.optim.RMSprop(
            params=self.critic.parameters(),
            lr=0.00025,
            alpha=0.95,
            centered=True,
            eps=0.01,
        )

        self.eps = 1.0
        self.min_eps = 0.1
        schedule_steps = 100_000
        self.ann_coeff = self.min_eps ** (1 / schedule_steps)

    def update(self, batches):
        for data in batches:
            data = jax.tree.map(crl.utils.to_torch, data)
            s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

            # Vanilla DQN
            q = self.critic(s).gather(-1, a.long())
            q_p = self.targ_critic(s_p).max(dim=-1, keepdim=True).values
            y = r + 0.99 * q_p * (1 - d)

            loss = F.smooth_l1_loss(q, y.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_count += 1
            if self.update_count % 1_000 == 0:
                self.targ_critic.load_state_dict(self.critic.state_dict())

    def step(self, state):
        if np.random.rand() > self.eps:
            th_state = th.from_numpy(state).unsqueeze(0).float()
            action = self.critic(th_state).argmax().detach().numpy()
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}


def main():
    def how_many_rollouts(env_steps, rollout_len, warmup_len):
        return int((env_steps - warmup_len) / rollout_len)

    env = gym.make("MinAtar/Freeway-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=DQN(env),
        capacity=100_000,
        rollout_len=1,
        batch_size=32,
        warmup_len=5_000,
    )

    n_steps = how_many_rollouts(
        env_steps=1_000_000,
        rollout_len=runner.rollout_len,
        warmup_len=runner.warmup_len,
    )

    runner.run(rollouts=n_steps)


if __name__ == "__main__":
    main()

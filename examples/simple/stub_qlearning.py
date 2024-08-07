import gymnasium as gym
import jax
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import cardio_rl as crl


class Q_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        q = self.net(state)
        return q


class DQN(crl.Agent):
    def __init__(self, env: gym.Env):
        self.env = env
        self.critic = Q_critic(4, 2)
        self.optimizer = th.optim.Adam(self.critic.parameters(), lr=7e-4)
        self.eps = 0.2

    def update(self, batch):
        data = jax.tree.map(crl.utils.to_torch, batch[0])
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        q = self.critic(s).gather(-1, a.unsqueeze(-1).long())

        q_p = self.critic(s_p).max(dim=-1, keepdim=True).values
        y = r + 0.99 * q_p * (1 - d.unsqueeze(-1))

        loss = F.mse_loss(q, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, state):
        if np.random.rand() > self.eps:
            th_state = th.from_numpy(state).unsqueeze(0).float()
            action = self.critic(th_state).argmax(-1).squeeze(0).detach().numpy()
        else:
            action = self.env.action_space.sample()
        return action, {}


def main():
    env = gym.make("CartPole-v1")
    runner = crl.BaseRunner(
        env=env,
        agent=DQN(env),
        rollout_len=32,
    )
    runner.run(rollouts=50_000)


if __name__ == "__main__":
    main()

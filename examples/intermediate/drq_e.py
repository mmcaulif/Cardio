"""
Data regularised Q from 'Image Augmentation Is All You Need: Regularizing
Deep Reinforcement Learning from Pixels' for discrete environments (Atari
100k).

Paper: https://arxiv.org/abs/2004.13649
Hyperparameters: page 17
Experiment details: page 7
Image augmentation details: page 22

DQN with double Q-learning, duellingn nets, n-step returns, tuned
hyperpameters and M/K random augmentations applied to S/S_p respectively.

Target networks are seemingly removed as target update period = 1.

To do:
* Image augmentation
* Benchmarking (Atari 100k)
* Review differences between DrQ and DrQ(e)
"""

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

        self.torso = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.v = nn.Linear(64, 1)
        self.a = nn.Linear(64, action_dim)

    def forward(self, state):
        z = self.torso(state)
        v = self.v(z)
        a = self.a(z)
        q = v + a - a.mean(-1, keepdim=True)
        return q


class DrQ(crl.Agent):
    def __init__(self, env: gym.Env, n_step: int):
        self.env = env
        self.n_step = n_step
        self.critic = Q_critic(4, 2)
        self.targ_critic = Q_critic(4, 2)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.update_count = 0
        self.optimizer = th.optim.Adam(
            params=self.critic.parameters(), lr=1e-4, eps=0.00015
        )

        self.eps = 1.0
        self.min_eps = 0.05
        schedule_steps = 5000
        self.ann_coeff = self.min_eps ** (1 / schedule_steps)

    def update(self, batches):
        data = jax.tree.map(crl.utils.to_torch, batches[0])
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        returns = th.zeros(r.shape[0])
        for i in reversed(range(r.shape[1])):
            returns += 0.99 * r[:, i]

        r = returns.unsqueeze(-1)

        q = self.critic(s).gather(-1, a.long())

        # If no targ critic why use double Q-learning???
        a_p = self.critic(s_p).argmax(-1, keepdim=True)
        q_p = self.targ_critic(s_p).gather(-1, a_p.long())
        y = r + np.power(0.99, self.n_step) * q_p * (1 - d)

        loss = F.mse_loss(q, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % 1 == 0:
            # Target update period == 1 so effectively no target net?
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
    env = gym.make("CartPole-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=DrQ(env, n_step=10),
        rollout_len=1,
        batch_size=32,
        warmup_len=1_600,
        n_step=10,
    )
    runner.run(rollouts=98_400)


if __name__ == "__main__":
    main()

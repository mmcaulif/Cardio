"""
Prioritized Experience Replay from 'Prioritized Experience
Replay' for discrete environments.

Paper:
Hyperparameters:
Experiment details:

DQN with double Q-learning and prioritised experience replay buffer

Notes:

To do:
* sanity check sumtree
"""

import copy

import gymnasium as gym
import jax
import numpy as np
import torch as th
import torch.nn as nn

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


class PER(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"lr": 1e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
    ):
        self.env = env
        self.critic = critic
        self.targ_critic = copy.deepcopy(critic)
        self.gamma = gamma
        self.targ_freq = targ_freq
        self.update_count = 0
        self.optimizer = th.optim.Adam(self.critic.parameters(), **optim_kwargs)

        self.eps = init_eps
        self.min_eps = min_eps
        self.ann_coeff = self.min_eps ** (1 / schedule_len)

    def update(self, batches):
        data = jax.tree.map(th.from_numpy, batches)
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        q = self.critic(s).gather(-1, a)
        a_p = self.critic(s_p).argmax(-1, keepdim=True)
        q_p = self.targ_critic(s_p).gather(-1, a_p)
        y = r + self.gamma * q_p * ~d
        error = q - y.detach()

        loss = th.mean((error**2) * data["w"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.targ_freq == 0:
            self.targ_critic.load_state_dict(self.critic.state_dict())

        return {"idxs": batches[0]["idxs"], "p": error.abs().numpy(force=True)}

    def step(self, state):
        if np.random.rand() > self.eps:
            th_state = th.from_numpy(state).unsqueeze(0).float()
            action = self.critic(th_state).argmax().detach().numpy()
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}

    def eval_step(self, state: np.ndarray):
        th_state = th.from_numpy(state)
        action = self.critic(th_state).argmax().numpy(force=True)
        return action


def main():
    env = gym.make("CartPole-v1")

    runner = crl.OffPolicyRunner(
        env,
        agent=PER(env, Q_critic(4, 2)),
        buffer=crl.buffers.PrioritisedBuffer(env),
        rollout_len=4,
        batch_size=32,
    )

    rollouts = 50_000
    runner.run(rollouts)


if __name__ == "__main__":
    main()

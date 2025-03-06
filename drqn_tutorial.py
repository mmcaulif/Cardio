"""Base implementation for Deep Recurrent Q-Networks with pytorch to be used as
a tutorial to demonstarte Cardio's extensiblity, as suggested by Pablo Samuel
Castro.

Once complete, move to a notebook.

Paper: https://arxiv.org/pdf/1507.06527
"""

import copy

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

        self.l1 = nn.Linear(state_dim, 32)
        self.gru = nn.GRU(32, 32)
        self.l2 = nn.Linear(32, action_dim)

    def forward(self, state, hx):
        z = F.relu(self.l1(state))
        z, hx = self.gru(z, hx)
        z = F.relu(z)
        q = self.l2(z)
        return q, hx


class DRQN(crl.Agent):
    def __init__(
        self,
        env: gym.Env,
        critic: nn.Module,
        gamma: float = 0.99,
        targ_freq: int = 1_000,
        optim_kwargs: dict = {"lr": 3e-4},
        init_eps: float = 0.9,
        min_eps: float = 0.05,
        schedule_len: int = 5000,
    ):
        self.env = env
        self.critic = critic
        self.targ_critic = copy.deepcopy(critic)
        self.hidden = th.zeros([1, 32])
        self.eval_hidden = th.zeros([1, 32])
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

        s = s.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        r = r.permute(1, 0, 2)
        s_p = s_p.permute(1, 0, 2)
        d = d.permute(1, 0, 2)

        _hx = th.zeros([1, s.shape[1], 32])

        q, _ = self.critic(s, _hx)
        q_p, _ = self.targ_critic(s_p, _hx)

        q = q.gather(-1, a)
        q_p = q_p.max(dim=-1, keepdim=True).values
        y = r + self.gamma * q_p * (1 - d)

        loss = F.mse_loss(q, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer.step()

        for targ_params, params in zip(
            self.targ_critic.parameters(), self.critic.parameters()
        ):
            targ_params.data.copy_(
                params.data * 0.005 + targ_params.data * (1.0 - 0.005)
            )

        return {}

    def step(self, state):
        if np.random.rand() > self.eps:
            th_state = th.from_numpy(state).unsqueeze(0)
            q_vals, self.hidden = self.critic(th_state, self.hidden)
            action = q_vals.argmax().numpy(force=True)
        else:
            th_state = th.from_numpy(state).unsqueeze(0)
            _, self.hidden = self.critic(th_state, self.hidden)
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}

    def eval_step(self, state: np.ndarray):
        th_state = th.from_numpy(state).unsqueeze(0)
        q_vals, self.eval_hidden = self.critic(th_state, self.eval_hidden)
        action = q_vals.argmax().numpy(force=True)
        return action

    def terminal(self):
        self.hidden = th.zeros([1, 32])

    def eval_terminal(self):
        self.eval_hidden = th.zeros([1, 32])


def main():
    env = gym.make("CartPole-v1")

    agent = DRQN(env, Q_critic(4, 2), schedule_len=100_000)

    runner = crl.Runner.off_policy(
        env=env,
        agent=agent,
        rollout_len=4,
        warmup_len=10_000,
        buffer_kwargs={"batch_size": 16, "trajectory": 6},
    )
    runner.run(rollouts=125_000, eval_freq=5_000)


if __name__ == "__main__":
    main()

import copy

import gymnasium as gym
import jax
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import cardio_rl as crl


class Q_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_critic, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        sa = th.concat([state, action], dim=-1)
        return self.net1(sa), self.net2(sa)

    def q1(self, state, action):
        sa = th.concat([state, action], dim=-1)
        return self.net1(sa)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, scale=1.0):
        super(Policy, self).__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        a = self.net(state)
        return th.mul(a, self.scale)


class TD3(crl.Agent):
    def __init__(self, env: gym.Env):
        self.env = env
        self.critic = Q_critic(3, 1)
        self.actor = Policy(3, 1)
        self.targ_critic = copy.deepcopy(self.critic)
        self.targ_actor = copy.deepcopy(self.actor)
        self.c_optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.a_optimizer = th.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.update_count = 0

    def update(self, batch):
        data = jax.tree.map(th.from_numpy, batch)
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        a_p = self.targ_actor(s_p)
        noise = th.normal(mean=th.zeros_like(a_p), std=0.2).clamp(-0.5, 0.5)
        a_p = (a_p + noise).clamp(-1.0, 1.0)

        q_p1, qp2 = self.targ_critic(s_p, a_p)
        q_p = th.min(q_p1, qp2)
        y = r + 0.98 * q_p * ~d

        q1, q2 = self.critic(s, a)

        loss = F.mse_loss(q1, y.detach()) + F.mse_loss(q2, y.detach())
        self.c_optimizer.zero_grad()
        loss.backward()
        self.c_optimizer.step()

        self.update_count += 1
        if self.update_count % 2 == 0:
            q1, q2 = self.critic(s, self.actor(s_p))
            policy_loss = -((q1 + q2) * 0.5).mean()

            self.a_optimizer.zero_grad()
            policy_loss.backward()
            self.a_optimizer.step()

            for targ_params, params in zip(
                self.targ_critic.parameters(), self.critic.parameters()
            ):
                targ_params.data.copy_(
                    params.data * 0.005 + targ_params.data * (1.0 - 0.005)
                )

            for targ_params, params in zip(
                self.targ_actor.parameters(), self.actor.parameters()
            ):
                targ_params.data.copy_(
                    params.data * 0.005 + targ_params.data * (1.0 - 0.005)
                )

        return {}

    def step(self, state):
        th_state = th.from_numpy(state)
        action = self.actor(th_state)
        noise = th.normal(mean=th.zeros_like(action), std=0.1).clamp(-0.5, 0.5)
        action = (action + noise).clamp(-1.0, 1.0)
        return action.numpy(force=True), {}

    def eval_step(self, state):
        th_state = th.from_numpy(state)
        action = self.actor(th_state).numpy(force=True)
        return action


def main():
    env = gym.make("Pendulum-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=TD3(env),
        rollout_len=1,
        batch_size=256,
    )
    runner.run(rollouts=190_000)


if __name__ == "__main__":
    main()

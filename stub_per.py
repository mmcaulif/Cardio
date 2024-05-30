import jax
import numpy as np
import cardio_rl as crl
import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self):
        self.critic = Q_critic(4, 2)
        self.targ_critic = Q_critic(4, 2)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.update_count = 0
        self.optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.eps = 0.9
        self.min_eps = 0.05
        schedule_steps = 5000
        self.ann_coeff = self.min_eps ** (1 / schedule_steps)

    def view(self, transition: dict, extras: dict):
        return {"error": 0.0}
        transition = jax.tree.map(crl.transitions.to_torch, transition)

        q = self.critic(transition["s"]).gather(0, transition["a"].long())

        a_p = self.critic(transition["s_p"]).argmax(dim=0, keepdim=True)
        q_p = self.targ_critic(transition["s_p"]).gather(0, a_p.long())
        y = transition["r"] + 0.99 * q_p * (1 - transition["d"])

        error = th.abs(q - y).detach().numpy() + 1e-8
        extras.update({"error": error})
        return extras

    def update(self, data):
        data = jax.tree.map(crl.transitions.to_torch, data)
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        q = self.critic(s).gather(1, a.long())

        a_p = self.critic(s_p).argmax(dim=1, keepdim=True)
        q_p = self.targ_critic(s_p).gather(1, a_p.long())
        y = r + 0.99 * q_p * (1 - d)

        loss = F.mse_loss(q, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % 1_000 == 0:
            self.targ_critic.load_state_dict(self.critic.state_dict())

    def step(self, state):
        if np.random.rand() > self.eps:
            input = th.from_numpy(state).unsqueeze(0).float()
            action = self.critic(input).argmax().detach().numpy()
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}


def main():
    runner = crl.OffPolicyRunner(
        env=gym.make("CartPole-v1"),
        agent=DQN(),
        extra_specs={"error": [1]},
        rollout_len=4,
        batch_size=32,
    )
    runner.run(50_000)


if __name__ == "__main__":
    main()

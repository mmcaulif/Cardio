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
    def __init__(self, env: gym.Env, n_step: int):
        self.env = env
        self.n_step = n_step
        self.critic = Q_critic(4, 2)
        self.targ_critic = Q_critic(4, 2)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.update_count = 0
        self.optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.eps = 0.9
        self.min_eps = 0.05
        schedule_steps = 5000
        self.ann_coeff = self.min_eps ** (1 / schedule_steps)

    def update(self, batches):
        for data in batches:
            data = jax.tree.map(crl.utils.to_torch, data)
            s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

            returns = th.zeros(r.shape[0])
            for i in reversed(range(r.shape[1])):
                returns += (0.99 * r[:, i])

            r = returns.unsqueeze(-1)

            q = self.critic(s).gather(-1, a.long())

            a_p = self.critic(s_p).argmax(-1, keepdim=True)
            q_p = self.targ_critic(s_p).gather(-1, a_p.long())
            y = r + np.power(0.99, self.n_step) * q_p * (1 - d)

            loss = F.mse_loss(q, y.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_count += 1
            if self.update_count % 1_000 == 0:
                self.targ_critic.load_state_dict(self.critic.state_dict())

    def _step(self, state):
        th_state = th.from_numpy(state).unsqueeze(0).float()
        action = self.critic(th_state).argmax().detach().numpy()
        return action

    def step(self, state):
        if np.random.rand() > self.eps:
            action = self._step(state)
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}
    
    def eval_step(self, state):
        return self._step(state), {}


def main():
    env = gym.make("CartPole-v1")
    runner = crl.OffPolicyRunner(
        env=env,
        agent=DQN(env, n_step=2),
        rollout_len=5,
        batch_size=32,
        n_step=2
    )
    runner.run(50_000, eval_interval=10_000, eval_episodes=1)


if __name__ == "__main__":
    main()

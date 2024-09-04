"""Rainbow from 'Rainbow: Combining Improvements in Deep Reinforcement
Learning' for discrete environments.

Paper:
Hyperparameters:
Experiment details:

DQN with double Q-learning, duelling Q-network, n-step returns,
prioritised experience replay, distributional Q-network, and noisy
networks.

Notes:
Implementation is based on Dopamine: PER beta value is fixed as 0.5,
instead of the linearly annealed original value. Additionally noisy
networks are left out of this implementation for now.

To do:
* C51
* Benchmarking (Atari 100k)
"""

import gymnasium as gym
import jax
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import cardio_rl as crl


class Q_critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_atoms=51):
        super(Q_critic, self).__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms

        self.torso = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.v = nn.Linear(64, n_atoms)
        self.a = nn.Linear(64, n_atoms * action_dim)

    def forward(self, state):
        z = self.torso(state)
        v = self.v(z).unsqueeze(-2)
        a = self.a(z)
        a = a.reshape(-1, self.action_dim, self.n_atoms)
        q = v + a - a.mean(-2, keepdim=True)
        return F.softmax(q, dim=-1)


class Rainbow(crl.Agent):
    def __init__(self, env: gym.Env, n_atoms: int = 51, beta: float = 0.5):
        self.env = env
        self.n_atoms = n_atoms
        self.beta = beta
        self.critic = Q_critic(4, 2, self.n_atoms)
        self.targ_critic = Q_critic(4, 2, self.n_atoms)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.update_count = 0
        self.optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.eps = 1.0
        self.min_eps = 0.05
        schedule_steps = 30_000
        self.ann_coeff = self.min_eps ** (1 / schedule_steps)

        self.v_min = 0
        self.v_max = 500
        self.bins = th.linspace(self.v_min, self.v_max, self.n_atoms)

    def update(self, batches):
        data = jax.tree.map(th.from_numpy, batches)
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        returns = th.zeros(r.shape[0])
        for i in reversed(range(r.shape[1])):
            returns += 0.99 * r[:, i]

        r = returns.unsqueeze(-1)

        # TODO: C51 loss function
        q = self.critic(s).gather(-1, a.repeat(1, self.n_atoms).unsqueeze(-2))

        dist_p = self.critic(s_p)
        a_p = (dist_p * self.bins).sum(-1).argmax(-1, keepdim=True)
        q_p = self.targ_critic(s_p).gather(
            -1, a_p.repeat(1, self.n_atoms).unsqueeze(-2)
        )
        print(q.shape, q_p.shape)

        raise NotImplementedError

        y = r + 0.99 * q_p * (1 - d)

        error = F.cross_entropy(q, y.detach())

        probs = data["p"] / th.sum(data["p"])
        w = probs**-self.beta  # equivelant to inverse square root of the probabilities
        w /= th.max(w)
        loss = th.mean((w * error) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % 1_000 == 0:
            self.targ_critic.load_state_dict(self.critic.state_dict())

        return {"idxs": batches[0]["idxs"], "p": np.abs(error.numpy(force=True))}

    def step(self, state):
        if np.random.rand() > self.eps:
            th_state = th.from_numpy(state).unsqueeze(0).float()
            dist = self.critic(th_state)
            action = (dist * self.bins).sum(-1).argmax().numpy(force=True)
        else:
            action = self.env.action_space.sample()

        self.eps = max(self.min_eps, self.eps * self.ann_coeff)
        return action, {}


def main():
    env = gym.make("CartPole-v1")
    agent = Rainbow(env)
    runner = crl.OffPolicyRunner(
        env,
        agent,
        buffer=crl.buffers.PrioritisedBuffer(env, n_steps=3),
        rollout_len=4,
        batch_size=32,
        n_step=3,
    )

    runner.run(50_000)


if __name__ == "__main__":
    main()

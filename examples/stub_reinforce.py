import gymnasium as gym
import jax
import torch as th
import torch.nn as nn

import cardio_rl as crl

"""
TODO: debug reinforce, it seems to plateau at an average return of ~40
"""


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(-1),
        )

    def forward(self, state):
        pi = self.net(state)
        return pi


class Reinforce(crl.Agent):
    def __init__(self):
        self.actor = Policy(4, 2)
        self.optimizer = th.optim.Adam(self.actor.parameters(), lr=1e-4)

    def update(self, batch):
        data = jax.tree.map(th.from_numpy, batch)
        s, a, r = data["s"], data["a"], data["r"]

        returns = th.zeros_like(r)

        rtg = 0.0
        for i in reversed(range(len(r))):
            rtg *= 0.99
            rtg += r[i]
            returns[i] = rtg

        probs = self.actor(s)
        dist = th.distributions.Categorical(probs)
        log_probs = dist.log_prob(a.squeeze())

        loss = th.mean(-log_probs * (returns - 100).squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, state):
        input_state = th.from_numpy(state)
        probs = self.actor(input_state)
        dist = th.distributions.Categorical(probs)
        action = dist.sample().squeeze()
        return action.numpy(force=True), {}


def main():
    runner = crl.BaseRunner(
        env=gym.make("CartPole-v1"), agent=Reinforce(), rollout_len=-1
    )
    runner.run(10_000, eval_episodes=100)


if __name__ == "__main__":
    main()

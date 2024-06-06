import gymnasium as gym
import jax
import torch as th
import torch.nn as nn

import cardio_rl as crl


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
        self.optimizer = th.optim.Adam(self.actor.parameters(), lr=3e-4)

    def update(self, data):
        data = jax.tree.map(crl.transitions.to_torch, data)
        s, a, r = data["s"], data["a"], data["r"]

        returns = th.zeros_like(r)

        rtg = 0
        for i in reversed(range(len(r))):
            rtg *= 0.99
            rtg += r[i]
            returns[i] = rtg

        probs = self.actor(s)
        dist = th.distributions.Categorical(probs)
        log_probs = dist.log_prob(a)

        loss = th.mean(-log_probs * (returns - 100))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, state):
        input_state = th.from_numpy(state).unsqueeze(0).float()
        probs = self.actor(input_state)
        dist = th.distributions.Categorical(probs)
        action = dist.sample().squeeze()
        log_probs = dist.log_prob(action).detach().numpy()
        return action.detach().numpy(), {"log_probs": log_probs}


def main():
    runner = crl.BaseRunner(
        env=gym.make("CartPole-v1"), agent=Reinforce(), rollout_len=-1, warmup_len=0
    )
    runner.run(10_000)


if __name__ == "__main__":
    main()

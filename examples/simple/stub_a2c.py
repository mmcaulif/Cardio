import gymnasium as gym
import jax
import torch as th
import torch.nn as nn

import cardio_rl as crl
from cardio_rl.types import Transition


class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        v = self.net(state)
        return v


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


class A2C(crl.Agent):
    def __init__(self):
        self.actor = Policy(4, 2)
        self.critic = Value(4)
        self.optimizer = th.optim.Adam(
            params=list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=3e-4,
        )

    def update(self, batches: list[Transition]):
        data = jax.tree.map(th.from_numpy, batches[0])
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        v = self.critic(s).squeeze(-1)
        v_p = self.critic(s_p).squeeze(-1)

        td_error = (
            # r + 0.99 * (v_p * (1 - d))    # TODO: Why does this not work for a2c?
            r + 0.99 * (v_p * ~d)
        ).detach() - v  # TODO: move to traditional adv calculation

        critic_loss = th.mean((td_error) ** 2)

        probs = self.actor(s)
        dist = th.distributions.Categorical(probs)
        log_probs = dist.log_prob(a)
        policy_loss = -th.mean(log_probs * td_error.detach())

        entropy = -dist.entropy().mean()

        loss = policy_loss + 0.5 * critic_loss + 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, state):
        input_state = th.from_numpy(state).unsqueeze(0).float()
        probs = self.actor(input_state)
        dist = th.distributions.Categorical(probs)
        action = dist.sample().squeeze()
        return action.numpy(force=True), {}

    def eval_step(self, state):
        input_state = th.from_numpy(state).unsqueeze(0).float()
        probs = self.actor(input_state)
        action = th.argmax(probs, dim=-1).squeeze()
        return action.numpy(force=True)


def main():
    envs = gym.make_vec("CartPole-v1", num_envs=16)
    eval_env = gym.make("CartPole-v1")

    runner = crl.BaseRunner(
        env=envs,
        agent=A2C(),
        rollout_len=32,
        gatherer=crl.VectorGatherer(),
        eval_env=eval_env,
    )

    runner.run(50_000, eval_freq=128)


if __name__ == "__main__":
    main()

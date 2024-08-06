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
        data = jax.tree.map(crl.utils.to_torch, batches[0])
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        v = self.critic(s).squeeze(-1)
        v_p = self.critic(s_p).squeeze(-1)

        td_error = (
            r + 0.99 * (v_p * (1 - d))
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
        return action.detach().numpy(), {}


def main():
    N_ENVS = 16
    envs = gym.make_vec("CartPole-v1", num_envs=N_ENVS)
    eval_env = gym.make("CartPole-v1")
    agent = A2C()

    runner = crl.BaseRunner(env=envs, rollout_len=32, gatherer=crl.VectorGatherer())

    for i in range(50_000):
        data = runner.step(agent=agent)
        agent.update(data)

        if i % 1_000 == 0 and i > 0:
            evals = 5
            returns = []
            for _ in range(evals):
                s, _ = eval_env.reset()
                R = 0.0
                while True:
                    a, _ = agent.step(s)
                    s, r, d, t, _ = eval_env.step(a)
                    R += r
                    if d or t:
                        returns.append(R)
                        break

            print(
                f"Reward at {i*N_ENVS*runner.rollout_len} steps: {sum(returns)/evals:.2f}"
            )
            returns.clear()


if __name__ == "__main__":
    main()

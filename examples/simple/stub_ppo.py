import gymnasium as gym
import jax
import numpy as np
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


class PPO(crl.Agent):
    def __init__(self, epochs, minibatches):
        self.actor = Policy(4, 2)
        self.critic = Value(4)
        self.optimizer = th.optim.Adam(
            params=list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=3e-4,
        )
        self.epochs = epochs
        self.minibatches = minibatches

    def update(self, batches: list[Transition]):
        # TODO: cleaner?
        data = jax.tree.map(crl.utils.to_torch, batches[0])
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        probs = self.actor(s)
        dist = th.distributions.Categorical(probs)
        old_log_probs = dist.log_prob(a).detach()

        batchsize = len(s)
        idxs = np.arange(batchsize)

        mb_size = batchsize // self.minibatches

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for n in range(self.minibatches):
                # TODO: add minibatching, look at CleanRL
                mb_idxs = idxs[n * mb_size : (n + 1) * mb_size]
                del mb_idxs

                v = self.critic(s).squeeze(-1)
                v_p = self.critic(s_p).squeeze(-1)

                td_error = (r + 0.99 * (v_p * (1 - d))).detach() - v

                gae = th.zeros_like(r)

                for i in reversed(range(len(r))):
                    if i == (len(r) - 1):
                        gae[i] = td_error[i]
                    else:
                        gae[i] = td_error[i] + (0.95 * 0.99 * gae[i + 1])

                returns = (gae + v_p).detach()  # TODO: double check this is correct
                critic_loss = th.mean((v - returns) ** 2)

                probs = self.actor(s)
                dist = th.distributions.Categorical(probs)
                log_probs = dist.log_prob(a)

                ratio = log_probs - old_log_probs  # TODO: check and add clipping
                policy_loss = -th.mean(
                    ratio * td_error.detach()
                )  # TODO: gae or td-error for policy adv?

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
    agent = PPO(2, 4)

    runner = crl.BaseRunner(
        env=envs, agent=agent, rollout_len=32, gatherer=crl.VectorGatherer()
    )

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
                    a, _ = runner.agent.step(s)
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

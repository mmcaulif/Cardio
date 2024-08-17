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
        data = jax.tree.map(th.from_numpy, batches)
        s, a, r, s_p, d = data["s"], data["a"], data["r"], data["s_p"], data["d"]

        probs = self.actor(s)
        dist = th.distributions.Categorical(probs)
        old_log_probs = dist.log_prob(a).detach()

        v = self.critic(s).squeeze(-1)
        v_p = self.critic(s_p).squeeze(-1)
        td_error = (r + 0.99 * (v_p * ~d)).detach() - v

        gae = th.zeros_like(r)

        for i in reversed(range(len(r))):
            if i == (len(r) - 1):
                gae[i] = td_error[i]
            else:
                gae[i] = td_error[i] + (0.95 * 0.99 * gae[i + 1])

        returns = (gae + v).detach()

        batchsize = len(s)
        idxs = np.arange(batchsize)
        mb_size = batchsize // self.minibatches

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for n in range(self.minibatches):
                mb_idxs = idxs[n * mb_size : (n + 1) * mb_size]

                new_v = self.critic(s[mb_idxs]).squeeze(-1)
                critic_loss = th.mean((new_v - returns[mb_idxs]) ** 2)

                probs = self.actor(s[mb_idxs])
                dist = th.distributions.Categorical(probs)
                log_probs = dist.log_prob(a[mb_idxs])

                ratio = log_probs - old_log_probs[mb_idxs]

                mb_gae = gae[mb_idxs].detach()

                policy_loss1 = -mb_gae * ratio
                policy_loss2 = -mb_gae * th.clamp(ratio, 1 - 0.2, 1 + 0.2)
                policy_loss = th.max(policy_loss1, policy_loss2).mean()

                entropy = -dist.entropy().mean()

                loss = policy_loss + 0.5 * critic_loss + 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def step(self, state):
        input_state = th.from_numpy(state)
        probs = self.actor(input_state)
        dist = th.distributions.Categorical(probs)
        action = dist.sample()
        return action.numpy(force=True), {}

    def eval_step(self, state):
        input_state = th.from_numpy(state)
        probs = self.actor(input_state)
        action = th.argmax(probs, dim=-1)
        return action.numpy(force=True)


def main():
    envs = gym.make_vec("CartPole-v1", num_envs=16)
    eval_env = gym.make("CartPole-v1")

    runner = crl.OnPolicyRunner(
        env=envs,
        agent=PPO(2, 4),
        rollout_len=32,
        gatherer=crl.VectorGatherer(),
        eval_env=eval_env,
    )

    runner.run(50_000, eval_freq=128, eval_episodes=20)


if __name__ == "__main__":
    main()

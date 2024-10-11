# type: ignore

from collections import deque

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ray.init()


@ray.remote
class MetricTracker:
    def __init__(self):
        self.window = deque(maxlen=400)
        self.episodes = 0

    def append(self, total_reward):
        self.window.append(total_reward)
        self.episodes += 1

        if self.episodes % 200 == 0:
            r_avg = sum(self.window) / len(self.window)
            print(f"Episodes completed: {self.episodes}, average reward: {r_avg}")


@ray.remote
class Runner:
    def __init__(self, env_name, rollout_len):
        self.env = gym.make(env_name)
        self.s, _ = self.env.reset()
        self.rollout_len = rollout_len
        self.r_sum = 0

    def rollout(self, model, logger):
        states = []
        actions = []
        rewards = []
        dones = []

        for _ in range(self.rollout_len):
            states.append(self.s)

            probs = model.policy(torch.tensor(self.s).float())
            dist = Categorical(probs=probs)

            a = dist.sample().numpy()
            actions.append(a)

            s_p, r, d, t, info = self.env.step(a)
            self.r_sum += r

            rewards.append(r)
            dones.append(d)
            self.s = s_p
            if d or t:
                logger.append.remote(self.r_sum)
                self.r_sum = 0
                self.s, _ = self.env.reset()

        value = model.critic(torch.tensor(self.s).float())

        return states, actions, rewards, dones, value


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.policy(state), self.critic(state)


# Initialisation
model = Model()
optim = torch.optim.AdamW(model.parameters(), lr=0.0007)  # , weight_decay=1e-4)

NUM_ACTORS = 8
NUM_STEPS = 1028

runners = [
    Runner.remote("CartPole-v1", NUM_STEPS // NUM_ACTORS) for _ in range(NUM_ACTORS)
]
logger = MetricTracker.remote()

for t in range(10_000):
    data_refs = [runner.rollout.remote(model, logger) for runner in runners]

    states = []
    actions = []
    rewards = []
    dones = []
    final_values = []

    while len(data_refs) > 0:
        ready, data_refs = ray.wait(data_refs, timeout=5.0)
        data = ray.get(ready)
        state, action, reward, done, last_val = data[0]

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        final_values.append(last_val)

    states = torch.from_numpy(np.array(states)).float()
    actions = torch.from_numpy(np.array(actions))
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones).int()
    final_values = torch.tensor(final_values).float()

    rtg = final_values
    returns = torch.zeros_like(rewards)

    for idx in reversed(range(NUM_STEPS // NUM_ACTORS)):
        rtg = (rewards[:, idx] + (0.99 * rtg)) * (1 - dones[:, idx])
        returns[:, idx] = rtg

    V = model.critic(states).squeeze(-1)

    c_loss = F.mse_loss(V, returns)

    adv = (returns - V).detach()

    probs = model.policy(states)
    log_probs = Categorical(probs=probs).log_prob(actions)

    pi_loss = -(log_probs * adv).mean()

    loss = pi_loss + (0.5 * c_loss)

    optim.zero_grad()
    loss.backward()
    optim.step()

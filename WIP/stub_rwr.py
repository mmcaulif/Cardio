from cardio_rl import Runner, Gatherer
import torch as th
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
import numpy as np


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.net(state)


env = gym.make("CartPole-v1")

runner = Runner(
    env=env,
    policy="categorical",
    collector=Gatherer(
        env=env,
        rollout_len=-1,
    ),
    backend="pytorch",
)

actor = Actor()
targ_actor = copy.deepcopy(actor)
critic = Critic()
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)

log_alpha = nn.Parameter(th.zeros(1))
alpha_optim = th.optim.Adam([log_alpha], lr=1e-3)
alpha = log_alpha.exp().detach().item()

for i in range(1_000):
    batch = runner.step(actor)
    s, a, r, s_p, d, _ = batch()

    returns = th.zeros_like(r)
    running_r = 0
    for t in reversed(range(len(r))):
        running_r = r[t] + 0.99 * running_r
        returns[t] = running_r

    adv: th.Tensor = (returns - 40).detach()

    weights = th.clamp_max(adv.exp(), 20)

    for _ in range(20):
        probs = actor(s)
        dist = Categorical(probs)

        targ_probs = targ_actor(s).detach()

        kl = th.mean((probs * th.log(probs / targ_probs)).sum(dim=-1))

        alpha_loss = -log_alpha.exp() * (0.01 - kl).detach().item()
        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

        alpha = log_alpha.exp().detach().item()

        policy_loss = -th.mean(dist.log_prob(a.squeeze(-1)) * (weights)) + (
            alpha * (0.01 - kl)
        )

        a_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.1)
        a_optimizer.step()

    targ_actor = copy.deepcopy(actor)

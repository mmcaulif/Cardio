import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Gatherer
from cardio_rl.policies import EpsilonArgmax

"""
A simple DQN implementation!
"""


class ImplicitQ(nn.Module):
    """
    Implicit Quantile Network: https://arxiv.org/abs/1806.06923
    -this helped a lot: https://datascience.stackexchange.com/questions/40874/how-does-implicit-quantile-regression-network-iqn-differ-from-qr-dqn
    -still need to implement quantile huber loss!
    """

    def __init__(self, state_dim, action_dim, emb_dim=64, n_samples=10):
        super(ImplicitQ, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)

        self.emb_dim = emb_dim
        self.embedding = nn.Linear(self.emb_dim, 128)
        self.n_samples = n_samples

    def forward(self, state, reduce=True):
        q = F.relu(self.l1(state))

        tau = th.zeros(self.n_samples, 1).uniform_()
        embedding = th.arange(self.emb_dim) * th.pi * tau
        embedding = th.cos(embedding)

        embedding = self.embedding(embedding)
        q = q.unsqueeze(-2) * embedding

        q = F.relu(self.l2(q))
        q = self.l3(q)

        if reduce:
            q = th.mean(q, dim=-2)

        return q


def main():
    env = gym.make("CartPole-v1")

    runner = Runner(
        env=env,
        policy=EpsilonArgmax(env, 0.5, 0.05, 0.9),
        sampler=True,
        capacity=100000,
        batch_size=64,
        collector=Gatherer(
            env=env,
            rollout_len=4,
            warmup_len=1000,
        ),
        backend="pytorch",
    )

    critic = ImplicitQ(4, 2)
    targ_net: nn.Module = copy.deepcopy(critic)
    optimizer = th.optim.Adam(critic.parameters(), lr=2.3e-3)
    gamma = 0.99
    target_update = 10

    for t in range(10000):
        batch = runner.step(critic)
        s, a, r, s_p, d, _ = batch()

        with th.no_grad():
            q_p = th.max(targ_net(s_p), keepdim=True, dim=-1).values
            y = r + gamma * q_p * (1 - d)

        q = critic(s).gather(1, a.long())

        loss = F.mse_loss(q, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % target_update == 0:
            targ_net = copy.deepcopy(critic)


if __name__ == "__main__":
    main()

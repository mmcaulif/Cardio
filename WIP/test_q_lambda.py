import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Gatherer


class Qfunc(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qfunc, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


def main():
    env = gym.make("CartPole-v1")

    runner = Runner(
        env=env,
        policy="argmax",
        sampler=True,
        capacity=1_000_000,
        batch_size=64,
        collector=Gatherer(env=env, rollout_len=4, warmup_len=1_000, n_step=4),
        reduce=False,
        backend="pytorch",
    )

    critic = Qfunc(4, 2)
    targ_net: nn.Module = copy.deepcopy(critic)
    optimizer = th.optim.Adam(critic.parameters(), lr=3e-4)
    gamma = 0.99
    target_update = 10

    for t in range(10000):
        batch = runner.step(critic)
        s, a, r, s_p, d, _ = batch()

        a = a.transpose(2, 1)
        r = r.transpose(2, 1)
        d = d.transpose(2, 1)

        """
		Q(lambda) implementation, need to look furtehr into watkins and pengs and the difference of the two
		"""

        lam = 0.8

        y_lam = th.zeros([runner.batch_size, 1])

        with th.no_grad():
            for l in range(r.shape[1]):
                n_rets = th.zeros([runner.batch_size, 1])

                # print(np.power(lam, l), r.shape[1] - l)

                for i in reversed(range(r.shape[1] - l)):
                    if i == (r.shape[1] - (1 + l)):
                        q_p = th.max(
                            targ_net(s_p[:, -(1 + l)]), keepdim=True, dim=-1
                        ).values
                        n_rets = r[:, i] + (0.99 * q_p) * (1 - d[:, i])
                    else:
                        n_rets = r[:, i] + (0.99 * n_rets) * (1 - d[:, i])

                y_lam += np.power(lam, l) * n_rets

        y_lam *= 1 - lam

        s = s[:, 0]
        a = a[:, 0]

        q = critic(s).gather(1, a.long())

        loss = F.mse_loss(q, y_lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % target_update == 0:
            targ_net = copy.deepcopy(critic)


if __name__ == "__main__":
    main()

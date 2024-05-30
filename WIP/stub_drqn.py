import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from cardio_rl import Runner
from cardio_rl import Gatherer
from cardio_rl.policies import EpsilonArgmax

# to impelment:
# -hidden state burn in
# -sequence overlap
# R2D2 paper for help: https://openreview.net/pdf?id=r1lyTjAqYX


class RecurrentQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RecurrentQ, self).__init__()
        self.gru = nn.GRU(state_dim, 64, batch_first=True)
        self.h1 = nn.Linear(64, 64)
        self.h2 = nn.Linear(64, action_dim)

    def forward(self, state, hidden, only_q=False):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)
        hidden, _ = self.gru(state, hidden)
        x = F.relu(hidden)
        x = F.relu(self.h1(x))
        q = self.h2(x)
        if only_q:
            return q
        return q, hidden


def main():
    env = gym.make("CartPole-v1")

    runner = Runner(
        env=env,
        policy=EpsilonArgmax(env, 0.5, 0.05, 0.9, True, 64),
        sampler=True,
        capacity=100_000,
        batch_size=256,
        collector=Gatherer(
            env=env,
            rollout_len=4,
            warmup_len=10_000,
            n_step=16,
            take_every=8,
        ),
        reduce=False,
        backend="pytorch",
    )

    critic = RecurrentQ(4, 2)
    targ_net: nn.Module = copy.deepcopy(critic)
    optimizer = th.optim.Adam(critic.parameters(), lr=1.0e-3)
    gamma = 0.99
    target_update = 10

    for t in range(10000):
        batch = runner.step(critic)
        s, a, r, s_p, d, _ = batch()

        # You are pretty close, just need to cross reference with this:
        # https://github.com/mynkpl1998/Recurrent-Deep-Q-Learning/blob/master/LSTM%2C%20BPTT%3D8.ipynb
        # also, implement the 'warm-start' use

        with th.no_grad():
            q_p = th.max(
                targ_net(s_p, th.zeros(1, runner.batch_size, 64), only_q=True),
                keepdim=True,
                dim=-1,
            ).values.transpose(1, 2)
            y = r + gamma * q_p * (1 - d)

        q = critic(s, th.zeros(1, runner.batch_size, 64), only_q=True).gather(
            2, a.long()
        )

        loss = F.mse_loss(q, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % target_update == 0:
            targ_net = copy.deepcopy(critic)


if __name__ == "__main__":
    main()

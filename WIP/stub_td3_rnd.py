import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from cardio_rl.policies import BasePolicy
from cardio_rl import Runner, Gatherer


class RndWhitenoiseDeterministic(BasePolicy):
    def __init__(self, env: gym.Env, obs_dims, output_dims):
        super().__init__(env)
        # architecture can maybe be improved on
        self.rnd_net = nn.Sequential(
            nn.Linear(obs_dims, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, output_dims),
        )

        self.targ_net = nn.Sequential(
            nn.Linear(obs_dims, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, output_dims),
        )

    def __call__(self, state, net):
        input = th.from_numpy(state).float()
        out = net(input)
        mean = th.zeros_like(out)
        noise = th.normal(mean=mean, std=0.1).clamp(-0.5, 0.5)
        out = out + noise
        return out.clamp(-1, 1).detach().numpy()

    def predictor(self, next_state):
        next_state = th.from_numpy(next_state).float()
        pred = self.rnd_net(next_state)
        targ = self.targ_net(next_state)
        intrinsic_r = (
            th.pow(th.mean(pred - targ, dim=-1, keepdim=True), 2).detach().item()
        )
        return intrinsic_r

    def update(self, rnd_net):
        self.rnd_net = rnd_net


class RndCollector(Gatherer):
    def _env_step(self, policy=None, warmup=False):
        if warmup:
            a = self.env.action_space.sample()
        else:
            a = policy(self.state, self.net)
        s_p, r, d, t, info = self.env.step(a)
        self.logger.step(r, d, t)
        d = d or t

        r_i = 1.0 * policy.predictor(s_p)
        r_e = r

        return (self.state, a, [r_e, r_i], s_p, d, info), s_p, d, t


env_name = "MountainCarContinuous-v0"
# env_name = 'Pendulum-v1'
# env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)
env = RescaleAction(env, -1.0, 1.0)

beta_rnd = 0.1

runner = Runner(
    env=env,
    policy=RndWhitenoiseDeterministic(env, obs_dims=2, output_dims=16),
    sampler=True,
    capacity=1_000_000,
    batch_size=512,
    n_batches=32,
    collector=RndCollector(
        env=env,
        rollout_len=32,
        warmup_len=0,
        logger_kwargs=dict(
            log_interval=5_000,
            episode_window=20,
            tensorboard=True,
            log_dir="run",
            exp_name=env_name + f"_RND_separate_nets_beta{beta_rnd}",
        ),
    ),
    backend="pytorch",
)


class Q_critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_critic, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.Dropout(0.01),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.01),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.Dropout(0.01),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.01),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action):
        sa = th.concat([state, action], dim=-1)
        return self.net1(sa), self.net2(sa)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
critic = Q_critic(2, 1)
int_critic = Q_critic(2, 1)
actor = Policy(2, 1)
targ_critic = copy.deepcopy(critic)
int_targ_critic = copy.deepcopy(int_critic)
targ_actor = copy.deepcopy(actor)
c_optimizer = th.optim.Adam(
    list(critic.parameters()) + list(int_critic.parameters()), lr=1e-3
)
a_optimizer = th.optim.Adam(actor.parameters(), lr=1e-3)
rnd_optimizer = th.optim.Adam(runner.policy.rnd_net.parameters(), lr=1e-3)

tau = 0.01

for steps in range(120_000):
    all_batches = runner.step(actor)
    for batch in all_batches:
        s, a, r, s_p, d, _ = batch()

        r_e, r_i = r[:, :, 0], r[:, :, 1]

        with th.no_grad():
            a_p = targ_actor(s_p)

            noise = th.normal(mean=th.zeros_like(a_p), std=0.2).clamp(-0.5, 0.5)
            a_p = (a_p + noise).clamp(-1, 1)

            q_p = th.min(*targ_critic(s_p, a_p))
            y = r_e + 0.9999 * q_p * (1 - d)

            i_q_p = th.min(*int_targ_critic(s_p, a_p))
            i_y = r_i + 0.99 * q_p * (1 - d)

        q1, q2 = critic(s, a.squeeze(1))
        i_q1, i_q2 = int_critic(s, a.squeeze(1))

        critic_loss = (F.mse_loss(q1, y) + F.mse_loss(q2, y)) + (
            F.mse_loss(i_q1, i_y) + F.mse_loss(i_q2, i_y)
        )

        c_optimizer.zero_grad()
        critic_loss.backward()
        c_optimizer.step()

        pred = runner.policy.rnd_net(s_p)
        targ = runner.policy.targ_net(s_p).detach()

        rnd_loss = F.mse_loss(pred, targ)

        rnd_optimizer.zero_grad()
        rnd_loss.backward()
        rnd_optimizer.step()

        if steps % 2 == 0:
            policy_loss = -(
                th.min(*critic(s_p, actor(s_p)))
                + beta_rnd * th.min(*int_critic(s_p, actor(s_p)))
            ).mean()

            a_optimizer.zero_grad()
            policy_loss.backward()
            a_optimizer.step()

            for targ_params, params in zip(
                targ_critic.parameters(), critic.parameters()
            ):
                targ_params.data.copy_(
                    params.data * tau + targ_params.data * (1.0 - tau)
                )

            for targ_params, params in zip(
                int_targ_critic.parameters(), int_critic.parameters()
            ):
                targ_params.data.copy_(
                    params.data * tau + targ_params.data * (1.0 - tau)
                )

            for targ_params, params in zip(targ_actor.parameters(), actor.parameters()):
                targ_params.data.copy_(
                    params.data * tau + targ_params.data * (1.0 - tau)
                )

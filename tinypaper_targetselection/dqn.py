import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from cardio_rl import Runner
from cardio_rl import Gatherer
from cardio_rl.policies import EpsilonArgmax
from cardio_rl.buffers.circular_buffer import ReplayBuffer

from tinypaper_targetselection.models import QNetConv, QNetMLP


def dqn_trial(cfg, env, logger_dict, grad_steps):
    runner = Runner(
        env=env,
        policy=EpsilonArgmax(env, 1.0, 0.1, 100_000),
        er_buffer=ReplayBuffer(env, cfg.buffer_size),
        batch_size=cfg.batch_size,
        collector=Gatherer(
            rollout_len=cfg.train_freq, warmup_len=cfg.warmup, logger_kwargs=logger_dict
        ),
        backend="pytorch",
    )

    if "MinAtar" in cfg.env_name:
        # Conv network and optimiser from MinAtar paper
        critic = QNetConv(env.game.state_shape()[2], 128, env.action_space.n)
        targ_critic: nn.Module = QNetConv(
            env.game.state_shape()[2], 128, env.action_space.n
        )
        targ_critic.load_state_dict(critic.state_dict())
        optimizer = th.optim.RMSprop(
            critic.parameters(),
            lr=cfg.learning_rate,
            alpha=0.95,
            centered=True,
            eps=0.01,
        )
    else:
        critic = QNetConv(env.observation_space.shape[0], 256, env.action_space.n)
        targ_critic: nn.Module = QNetConv(
            env.observation_space.shape[0], 64, env.action_space.n
        )
        targ_critic.load_state_dict(critic.state_dict())
        optimizer = None

    for t in trange(grad_steps):
        if cfg.target_selection:
            batch = runner.step(targ_critic)
        else:
            batch = runner.step(critic)

        s, a, r, s_p, d = batch

        s, s_p = s.squeeze(1), s_p.squeeze(1)

        q = critic(s).gather(1, a.long())

        with th.no_grad():
            # Double DQN
            a_p = critic(s_p).argmax(dim=1).unsqueeze(1)
            q_p = targ_critic(s_p).gather(1, a_p.long())
            y = r + cfg.gamma * q_p * (1 - d)

        loss = F.mse_loss(q, y)

        optimizer.zero_grad()
        loss.backward()
        # th.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optimizer.step()

        if cfg.target_update >= 1.0 and t % cfg.target_update == 0:
            targ_critic.load_state_dict(critic.state_dict())

        elif 1.0 > cfg.target_update:
            for targ_params, params in zip(
                targ_critic.parameters(), critic.parameters()
            ):
                targ_params.data.copy_(
                    params.data * cfg.target_update
                    + targ_params.data * (1.0 - cfg.target_update)
                )

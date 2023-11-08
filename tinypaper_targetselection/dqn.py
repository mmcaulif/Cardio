import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from cardio_rl import Runner
from cardio_rl import Collector
from cardio_rl.policies import EpsilonArgmax

from tinypaper_targetselection.models import QNetConv, QNetMLP


def dqn_trial(cfg, env, grad_steps):

	runner = Runner(
		env=env,
		policy=EpsilonArgmax(env, 1.0, 0.1, 0.999954),	# 0.999954 reaches 0.1 after ~100,000 timesteps
		batch_size=cfg.batch_size,
		collector=Collector(
			rollout_len=cfg.train_freq,
			warmup_len=cfg.warmup,
			logger_kwargs=dict(
				tensorboard=cfg.tensorboard,
				log_dir='tb_logs/',
				exp_name=f'{cfg.env_name.replace("/", "-")}_targsel{cfg.target_selection}'
				)
		),
		backend='pytorch'
	)

	critic = QNetConv(env.game.state_shape()[2], 128, env.action_space.n)
	targ_critic: nn.Module = copy.deepcopy(critic)

	if 'MinAtar' in cfg.env_name:
		# Optimiser from MinAtar paper
		optimizer = th.optim.RMSprop(critic.parameters(), lr=cfg.learning_rate, alpha=0.95, centered=True, eps=0.01)
	else:
		optimizer = None

	for t in trange(grad_steps):

		if cfg.target_selection:
			batch = runner.get_batch(targ_critic)


		s, a, r, s_p, d, _ = batch

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
		th.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
		optimizer.step()

		if t % cfg.target_update == 0:        
			targ_critic = copy.deepcopy(critic)

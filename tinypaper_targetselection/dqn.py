import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from tqdm import trange
from gymnasium.wrappers import TransformObservation
from cardio_rl import Runner
from cardio_rl import Collector
from cardio_rl.policies import EpsilonArgmax

from tinypaper_targetselection.models import QNetConv, QNetMLP

"""
Small code for tiny paper submission for target network action selection to combat policy churn
Example DQN: https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
MinAtar paper: https://arxiv.org/pdf/1903.03176.pdf
[x] Install MinAtar 
[x] Setup conv net architecture for Q-func
[x] List hyperparams
[ ] Create config!
[ ] Run experiments!

Next:
[ ] Do the same for SAC and TD3
"""

############################
# 	Constants
############################
# BATCH_SIZE = 32
# REPLAY_BUFFER_SIZE = 100000
# TARGET_NETWORK_UPDATE_FREQ = 1000
# TRAINING_FREQ = 1
# NUM_FRAMES = 5000000
# FIRST_N_FRAMES = 100000
# REPLAY_START_SIZE = 5000
# END_EPSILON = 0.1
# STEP_SIZE = 0.00025
# GRAD_MOMENTUM = 0.95
# SQUARED_GRAD_MOMENTUM = 0.95
# MIN_SQUARED_GRAD = 0.01
# GAMMA = 0.99
# EPSILON = 1.0


@hydra.main(version_base=None, config_path='configs', config_name='main.yaml')
def main(cfg):
	rprint(OmegaConf.to_yaml(cfg))
	exit()
	for _ in range(1):
		trial(cfg)

def trial(cfg, grad_steps=5_000_000):
	env = gym.make('MinAtar/Freeway-v1')
	extract_obs = lambda s: (th.from_numpy(s).permute(2, 0, 1)).unsqueeze(0).float().detach().numpy()
	env = TransformObservation(env, extract_obs)

	runner = Runner(
		env=env,
		policy=EpsilonArgmax(env, 1.0, 0.1, 0.999954),	# 0.999954 reaches 0.1 after ~100,000 timesteps
		batch_size=32,
		collector=Collector(
			rollout_len=1,
			warmup_len=5000,
		),
		backend='pytorch'
	)

	critic = QNetConv(env.game.state_shape()[2], 128, env.action_space.n)
	targ_net: nn.Module = copy.deepcopy(critic)
	optimizer = th.optim.RMSprop(critic.parameters(), lr=0.00025, alpha=0.95, centered=True, eps=0.01)
	gamma = 0.99
	target_update = 1000

	for t in trange(grad_steps):
		batch = runner.get_batch(critic)
		s, a, r, s_p, d, _ = batch

		s, s_p = s.squeeze(1), s_p.squeeze(1)

		q = critic(s).gather(1, a.long())

		with th.no_grad():
			a_p = critic(s_p).argmax(dim=1).unsqueeze(1)		
			q_p = targ_net(s_p).gather(1, a_p.long())		
			y = r + gamma * q_p * (1 - d)

		loss = F.mse_loss(q, y)

		optimizer.zero_grad()
		loss.backward()
		th.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
		optimizer.step()

		if t % target_update == 0:        
			targ_net = copy.deepcopy(critic)
			
if __name__ == '__main__':
	main()
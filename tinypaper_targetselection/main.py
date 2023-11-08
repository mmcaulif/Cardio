import torch as th
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from gymnasium.wrappers import TransformObservation
from dqn import dqn_trial

"""
Small code for tiny paper submission for target network action selection to combat policy churn
Example DQN: https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
MinAtar paper: https://arxiv.org/pdf/1903.03176.pdf
[x] Install MinAtar 
[x] Setup conv net architecture for Q-func
[x] List hyperparams
[ ] Create config (base config + override configs per algorithm + environment (1 for MinAtar, 1 for LunarLander))!
[ ] Decoupel main and trial function
[ ] Run experiments (DQN first)!

Next:
[ ] Do the same for SAC and TD3
"""

ALGOS = {'DQN': dqn_trial}

@hydra.main(version_base=None, config_path='configs', config_name='main.yaml')
def main(cfg):
	rprint(OmegaConf.to_yaml(cfg))

	env = gym.make(cfg.env_name, disable_env_checker=True)

	if 'MinAtar' in cfg.env_name:
		extract_obs = lambda s: (th.from_numpy(s).permute(2, 0, 1)).unsqueeze(0).float().detach().numpy()
		env = TransformObservation(env, extract_obs)

	grad_steps = ((cfg.env_steps - cfg.warmup)// cfg.train_freq)

	trial = ALGOS[cfg.algorithm]

	exit()

	for _ in range(cfg.n_trials):
		trial(cfg, env, grad_steps)
		
			
if __name__ == '__main__':
	main()

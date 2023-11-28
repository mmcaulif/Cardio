import torch as th
import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from gymnasium.wrappers import TransformObservation
from dqn import dqn_trial
from wrapper import TransformObsWrapper

ALGOS = {'DQN': dqn_trial}

@hydra.main(version_base=None, config_path='configs', config_name='main.yaml')
def main(cfg):
	rprint(OmegaConf.to_yaml(cfg))

	grad_steps = ((cfg.exp.env_steps - cfg.alg.warmup)// cfg.alg.train_freq)

	trial = ALGOS[cfg.alg.algorithm]

	for i in range(cfg.exp.n_trials):

		env = gym.make(cfg.alg.env_name, disable_env_checker=True)

		if 'MinAtar' in cfg.alg.env_name:
			env = TransformObsWrapper(env)

		logger_dict = dict(
			log_interval=10_000,
			tensorboard=cfg.exp.tensorboard,
			log_dir=f'tb_logs/{cfg.exp.exp_dir}/{cfg.alg.env_name.replace("/", "-")}/',
			exp_name=f'{cfg.exp.name}_targsel_{cfg.alg.target_selection}_{i+1}'
		)
		
		trial(cfg.alg, env, logger_dict, grad_steps)
		
			
if __name__ == '__main__':
	main()

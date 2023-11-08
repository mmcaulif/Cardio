import torch as th
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from gymnasium.wrappers import TransformObservation
from dqn import dqn_trial


ALGOS = {'DQN': dqn_trial}

@hydra.main(version_base=None, config_path='configs', config_name='main.yaml')
def main(cfg):
	rprint(OmegaConf.to_yaml(cfg))

	grad_steps = ((cfg.exp.env_steps - cfg.alg.warmup)// cfg.alg.train_freq)

	trial = ALGOS[cfg.alg.algorithm]

	for i in range(cfg.exp.n_trials):

		env = gym.make(cfg.alg.env_name, disable_env_checker=True)

		if 'MinAtar' in cfg.alg.env_name:
			extract_obs = lambda s: (th.from_numpy(s).permute(2, 0, 1)).unsqueeze(0).float().detach().numpy()
			env = TransformObservation(env, extract_obs)

		logger_dict = dict(
			tensorboard=cfg.exp.tensorboard,
			log_dir=f'tb_logs/{cfg.alg.env_name.replace("/", "-")}/',
			exp_name=f'targsel_{cfg.alg.target_selection}_{i+1}'
		)
		
		trial(cfg.alg, env, logger_dict, grad_steps)
		
			
if __name__ == '__main__':
	main()

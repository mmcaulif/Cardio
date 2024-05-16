from typing import Any
import jax
import numpy as np
import cardio_rl as crl
import gymnasium as gym
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import rlax

from flax.training.train_state import TrainState

class DdqnTrainState(TrainState):
	target_params: flax.core.FrozenDict[str, Any]

class Q_critic(nn.Module):
	action_dim: int
	@nn.compact
	def __call__(self, s):
		z = nn.relu(nn.Dense(64)(s))
		z = nn.relu(nn.Dense(64)(z))
		z = nn.Dense(self.action_dim)(z)
		return z

class DQN(crl.Agent):
	def __init__(self):
		self.key = jax.random.PRNGKey(0)
		self.key, init_key = jax.random.split(self.key)
		model = Q_critic(action_dim=2)
		params = model.init(init_key, jnp.zeros(4))
		optim = optax.adam(1e-4)
		opt_state = optim.init(params)

		self.train_state = DdqnTrainState(
			step=0,
			apply_fn=jax.jit(model.apply),
			params=params,
			opt_state=opt_state,
			tx=optim,
			target_params=params
		)

		self.eps = 0.9
		self.min_eps = 0.05
		schedule_steps = 5000 
		self.ann_coeff = self.min_eps ** (1/schedule_steps)
		
	def update(self, data):
		
		@jax.jit
		def _update(train_state: DdqnTrainState, data: dict):

			def _loss(params, data: dict, train_state: DdqnTrainState):
				q = train_state.apply_fn(params, data['s'])
				q_p_value = train_state.apply_fn(train_state.target_params, data['s_p'])
				q_p_select = train_state.apply_fn(train_state.target_params, data['s_p'])
				discount = (1 - data['d']) * 0.99
				error = jax.vmap(rlax.double_q_learning)(
					q, 
					jnp.squeeze(data['a']), 
					jnp.squeeze(data['r']), 
					jnp.squeeze(discount), 
					q_p_value,
					q_p_select
				)
				return jnp.mean(error**2)

			grads = jax.grad(_loss)(train_state.params, data, train_state)

			train_state = train_state.apply_gradients(grads=grads)

			target_params = optax.periodic_update(
				train_state.params,
				train_state.target_params,
				train_state.step,
				1_000
			)

			train_state = train_state.replace(target_params=target_params)
			return train_state

		new_train_state = _update(
			self.train_state, 
			data
		)

		self.train_state = new_train_state

	def step(self, state):

		if np.random.rand() > self.eps:
			q = self.train_state.apply_fn(self.train_state.params, state)
			action = np.array(q.argmax())
		else:
			action =  self.env.action_space.sample()
		
		self.eps = max(self.min_eps, self.eps*self.ann_coeff)
		return action, {}


def main():
	runner = crl.OffPolicyRunner(
		env=gym.make("CartPole-v1"),
		agent=DQN(),
		rollout_len=4,
		batch_size=32,
	)
	runner.run(10_000)

if __name__ == '__main__':
	main()
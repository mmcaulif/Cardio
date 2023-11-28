import copy
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import distrax
import gymnasium as gym

from cardio_rl import Runner, Gatherer

env = gym.make('Pendulum-v1')

runner = Runner(
	env=env,
	policy='random',	# need to implement jax policies
	sampler=True,
	capacity=100000,
	batch_size=256,
	collector=Gatherer(
		env=env,
		warmup_len=100,
	),
	backend='jax'
)

class Actor(nn.Module):
	@nn.compact
	def __call__(self, state):
		x = nn.Dense(128)(state)
		x = nn.relu(x)

		# Deterministic policy for DDPG
		x = nn.Dense(128)(x)
		x = nn.relu(x)
		x = nn.Dense(1)(x)
		x = nn.relu(x)
		return x

		# Beta policy for SAC
		"""alpha = nn.Dense(128)(x)
		alpha = nn.relu(alpha)
		alpha = nn.Dense(1)(alpha)
		alpha = nn.softplus(alpha)

		beta = nn.Dense(128)(x)
		beta = nn.relu(beta)
		beta = nn.Dense(1)(beta)
		beta = nn.softplus(beta)
		return alpha, beta"""

class Critic(nn.Module):
	@nn.compact
	def __call__(self, state, action):
		sa = jnp.concatenate([state, action], axis=-1)
		x = nn.Dense(128)(sa)
		x = nn.relu(x)
		x = nn.Dense(128)(x)
		x = nn.relu(x)
		x = nn.Dense(1)(x)
		return x


actor = Actor()
a_batch = jnp.ones(3)
a_params = actor.init(jax.random.PRNGKey(0), a_batch)
a_optimizer = optax.adam(learning_rate=1e-3)
a_opt_state = a_optimizer.init(a_params)
output = actor.apply(a_params, a_batch)

critic = Critic()
c_batch = (jnp.ones(3), jnp.ones(1))
c_params = critic.init(jax.random.PRNGKey(0), *c_batch)
targ_c_params = copy.deepcopy(c_params)
c_optimizer = optax.adam(learning_rate=1e-3)
c_opt_state = a_optimizer.init(c_params)
output = critic.apply(c_params, *c_batch)

ent_coeff = 0.1

for steps in range(20_000):
	if steps % 100 == 0 and steps > 0:
		print(steps, runner.policy)

	batch = runner.step(actor)

	# first step is implementing Transition/backend selection
	s, a, r, s_p, d = batch()

	a_p = actor.apply(a_params, s_p)
	q_p = critic.apply(targ_c_params, s_p, a_p)

	"""
	alpha, beta = actor.apply(params, x)
	dist = distrax.Beta(alpha, beta)
	a_p = (dist.sample(jax.random.PRNGKey(0)) * 4) - 2
	return -jnp.mean(critic.apply(q_params, x, a_p))
	q_p = critic.apply(targ_c_params, s_p, a_p) - ent_coeff*dist.entropy()
	y = jax.lax.stop_gradient(r + 0.99 *  q_p * (1 - d))	
	"""

	y = jax.lax.stop_gradient(r + 0.99 * q_p * (1 - d))	

	@jax.jit
	@jax.value_and_grad
	def td_error(params, y):
		q = critic.apply(params, s, a)
		return jnp.mean(optax.l2_loss(q, y))	
	
	c_loss, c_grads = td_error(c_params, y)
	updates, c_opt_state = c_optimizer.update(c_grads, c_opt_state)
	c_params = optax.apply_updates(c_params, updates)

	if steps % 2 == 0:
		@jax.jit
		@jax.value_and_grad
		def policy_loss(params, q_params, x):
			a_sampled = actor.apply(params, x)
			return -jnp.mean(critic.apply(q_params, x, a_sampled))

		a_loss, a_grads = policy_loss(a_params, c_params, s)
		updates, a_opt_state, = a_optimizer.update(a_grads, a_opt_state)
		a_params = optax.apply_updates(a_params, updates)

		continue
		@jax.jit
		@jax.value_and_grad
		def policy_loss(params, q_params, x):
			alpha, beta = actor.apply(params, x)
			dist = distrax.Beta(alpha, beta)
			a_sampled = (dist.sample(jax.random.PRNGKey(0)) * 4) - 2
			return -jnp.mean(critic.apply(q_params, x, a_sampled) - (ent_coeff*dist.entropy()))
		

	# see if you can do this more smoothly
	targ_c_params = jax.tree_util.tree_map(lambda x, y: (1 - 0.005) * x + 0.005 * y, targ_c_params, c_params)
	
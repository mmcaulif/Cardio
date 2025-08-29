import jax
from mujoco_playground import registry, wrapper

# env_name = 'Go1JoystickFlatTerrain'
# env = registry.load(env_name)
# key = jax.random.PRNGKey(0)
# state = env.reset(key)
# print(state.obs['state'])
# print(state.obs['privileged_state'])
# print(state.obs.keys())

env = registry.load('CartpoleBalance')
key = jax.random.PRNGKey(0)
state = env.reset(key)
print(state.obs)
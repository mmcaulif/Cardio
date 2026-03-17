from brax import envs

env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'spring'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name, backend=backend)
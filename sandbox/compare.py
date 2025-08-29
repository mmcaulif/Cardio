import chex
import gymnax
import jax
import jax.numpy as jnp


def repeat_state(state, n):
    """Repeat the state for n parallel environments."""
    return jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, axis=0), state)


def broadcast_tree(struct: chex.ArrayTree, add_dims: tuple[int], axis: int = 0) -> chex.ArrayTree:
    """Add dimensions to each array in a tree structure and broadcast values along those dimensions.
    This function takes a PyTree of arrays and adds additional dimensions at the specified axis,
    then broadcasts the original values across the newly added dimensions.
    Args:
        struct: A PyTree of arrays to be broadcasted.
        add_dims: A tuple specifying the dimensions to add at the specified axis.
        axis: The axis position where new dimensions should be inserted. Default is 0.
    Returns:
        A new PyTree with the same structure but with arrays that have been expanded
        and broadcasted along the new dimensions.
    """

    def broadcast_fn(x: chex.Array) -> chex.Array:
        x_shape = x.shape
        prefix = x_shape[:axis]
        suffix = x_shape[axis:]
        new_shape = prefix + add_dims + suffix
        x = jnp.expand_dims(x, axis=axis)
        return jnp.broadcast_to(x, new_shape)

    return jax.tree_util.tree_map(
        broadcast_fn,
        jax.tree_util.tree_map(jnp.asarray, struct),
    )


N_PARTICLES = 2

key = jax.random.key(0)
key, key_reset, key_step = jax.random.split(key, 3)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("SpaceInvaders-MinAtar")
obs, state = env.reset(key_reset, env_params)

repeated_state = repeat_state(state, N_PARTICLES)
broadcasted_state = broadcast_tree(repeated_state, (N_PARTICLES,))

# print("Repeated state:", repeated_state)
print("Repeated state:", jax.tree.map(lambda x: x.shape, repeated_state))
print()
# print("Broadcasted state:", broadcasted_state)
print("Broadcasted state:", jax.tree.map(lambda x: x.shape, broadcasted_state))

# print(jax.tree.map(lambda x, y: jnp.array_equal(x, y), repeated_state, broadcasted_state))

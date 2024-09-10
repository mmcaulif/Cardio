import flax.linen as nn
import jax.debug
import jax.numpy as jnp


class NoisyDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, key, eval=False):
        def mu_init(key, shape):
            low = -3 / jnp.sqrt(x.shape[0])
            high = 3 / jnp.sqrt(x.shape[0])
            return jax.random.uniform(key, minval=low, maxval=high, shape=shape)

        def sigma_init(key, shape, dtype=jnp.float32):
            return jnp.ones(shape, dtype) * 0.017

        def f(x):
            return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

        kernel_mu = self.param(
            "kernel_mu",
            mu_init,
            (x.shape[-1], self.features),
        )

        kernel_sigma = self.param(
            "kernel_sigma",
            sigma_init,
            (x.shape[-1], self.features),
        )

        bias_mu = self.param(
            "bias_mu",
            mu_init,
            (self.features,),
        )
        bias_sigma = self.param(
            "bias_sigma",
            sigma_init,
            (self.features,),
        )

        p_key, q_key = jax.random.split(key)  # type: ignore

        # Factorised noise
        p = jax.random.normal(p_key, [x.shape[0], 1])
        q = jax.random.normal(q_key, [1, self.features])
        f_p = f(p)
        f_q = f(q)
        w_eps = f_p * f_q
        b_eps = jnp.squeeze(f_q)

        # Non-factorised noise
        # w_eps = jax.random.normal(p_key, (x.shape[0], self.features,))
        # b_eps = jax.random.normal(q_key, (self.features,))

        kernel = (w_eps * kernel_sigma) * (1 - eval) + kernel_mu

        y = jnp.matmul(x, kernel)

        bias = (b_eps * bias_sigma) * (1 - eval) + bias_mu

        y += bias
        return y

import flax.linen as nn
import jax
import jax.numpy as jnp


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    num_heads: int = 4
    emb_dim: int = 128
    @nn.compact
    def __call__(self, x):
        """Forward pass for multi-head self-attention."""
        local_emb_dim = self.emb_dim // self.num_heads
        q = nn.DenseGeneral((self.num_heads, local_emb_dim))(x)
        k = nn.DenseGeneral((self.num_heads, local_emb_dim))(x)
        v = nn.DenseGeneral((self.num_heads, local_emb_dim))(x)
        k_t = jnp.transpose(k, (0, 2, 1))
        weights = nn.softmax(q @ k_t / jnp.sqrt(local_emb_dim))
        return jnp.reshape(weights @ v, (x.shape[0], self.emb_dim))


class Block(nn.Module):
    """A block of multi-head self-attention followed by a feed-forward network."""
    num_heads: int = 4
    emb_dim: int = 128

    @nn.compact
    def __call__(self, x):
        """Forward pass for the block."""
        attn = MultiheadSelfAttention(num_heads=self.num_heads, emb_dim=self.emb_dim)(x)
        x = nn.LayerNorm()(x + attn)  # Residual connection and layer normalization
        x = nn.Dense(self.emb_dim)(x)  # Feed-forward network
        return nn.LayerNorm()(x + attn)  # Another residual connection and layer normalization

class Encoder(nn.Module):
    """Encoder that applies multiple blocks of self-attention."""
    vocab_size: int = 1000  # Example vocabulary size
    num_blocks: int = 4
    num_heads: int = 4
    emb_dim: int = 128

    @nn.compact
    def __call__(self, x):
        """Forward pass for the encoder."""
        embeddings = nn.Embed(num_embeddings=self.vocab_size, features=self.emb_dim)(x)  # Example embedding layer
        for _ in range(self.num_blocks):
            embeddings = Block(num_heads=self.num_heads, emb_dim=self.emb_dim)(embeddings)
        logits = nn.Dense(self.vocab_size)(embeddings)
        return logits

net = Encoder()

x = jax.random.randint(jax.random.PRNGKey(0), (10,), 0, 1000)  # Example input of shape (10,) with vocab size 1000
params = net.init(jax.random.PRNGKey(0), x)
y = net.apply(params, x)
print(y.shape)  # Should print (10, 1000) since the output is the same shape as the input

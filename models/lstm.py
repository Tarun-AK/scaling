from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class LSTMLanguageModel(nn.Module):
    hidden_dim: int
    num_layers: int
    vocab_size: int
    embed_dim: int | None = None

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        batch_size = tokens.shape[0]
        embed_dim = self.hidden_dim if self.embed_dim is None else self.embed_dim

        x = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim, name="embed")(
            tokens
        )
        # x shape: (batch, seq_len, embed_dim)

        for layer_idx in range(self.num_layers):
            x = nn.RNN(nn.LSTMCell(self.hidden_dim), name=f"rnn_{layer_idx}")(x)
            # x shape: (batch, seq_len, hidden_dim)

        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
        return logits

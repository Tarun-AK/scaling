from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class LSTMLanguageModel(nn.Module):
    hidden_dim: int
    num_layers: int
    vocab_size: int
    embed_dim: int | None = None

    def init_carry(self, batch_size: int) -> tuple:
        """Initialize LSTM carry for each layer."""
        carry = []
        for _ in range(self.num_layers):
            c = jnp.zeros((batch_size, self.hidden_dim))
            h = jnp.zeros((batch_size, self.hidden_dim))
            carry.append((c, h))
        return tuple(carry)

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

    @nn.compact
    def step(self, carry: tuple, token: jax.Array) -> tuple:
        """Single LSTM step for autoregressive sampling.

        Args:
            carry: LSTM carry (tuple of (c, h) per layer)
            token: (batch,) integer array of a single token

        Returns:
            (new_carry, logits) where logits is (batch, vocab_size)
        """
        embed_dim = self.hidden_dim if self.embed_dim is None else self.embed_dim

        x = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim, name="embed")(
            token
        )

        class _StepCell(nn.Module):
            hidden_dim: int

            @nn.compact
            def __call__(self, cell_carry: tuple, inputs: jax.Array) -> tuple:
                lstm_cell = nn.LSTMCell(self.hidden_dim, name="cell")
                return lstm_cell(cell_carry, inputs)

        new_carry = []
        for layer_idx in range(self.num_layers):
            step_cell = _StepCell(self.hidden_dim, name=f"rnn_{layer_idx}")
            layer_carry = carry[layer_idx] if carry else None
            new_layer_carry, x = step_cell(layer_carry, x)
            new_carry.append(new_layer_carry)

        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
        return (tuple(new_carry), logits)

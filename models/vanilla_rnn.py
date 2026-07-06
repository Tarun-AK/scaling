from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class VanillaRNNLanguageModel(nn.Module):
    hidden_dim: int
    num_layers: int
    vocab_size: int
    embed_dim: int | None = None

    def init_carry(self, batch_size: int) -> tuple[jax.Array, ...]:
        carry = []
        for _ in range(self.num_layers):
            h = jnp.zeros((batch_size, self.hidden_dim))
            carry.append(h)
        return tuple(carry)

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        embed_dim = self.hidden_dim if self.embed_dim is None else self.embed_dim
        x = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim, name="embed")(
            tokens
        )

        for layer_idx in range(self.num_layers):
            x = nn.RNN(nn.SimpleCell(self.hidden_dim), name=f"rnn_{layer_idx}")(x)

        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
        return logits

    @nn.compact
    def step(
        self,
        carry: tuple[jax.Array, ...],
        token: jax.Array,
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        embed_dim = self.hidden_dim if self.embed_dim is None else self.embed_dim
        x = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim, name="embed")(
            token
        )

        new_carry = []
        for layer_idx in range(self.num_layers):
            step_cell = nn.SimpleCell(
                self.hidden_dim,
                name=f"SimpleCell_{layer_idx}",
            )
            h_prev = carry[layer_idx]
            h_t, x = step_cell(h_prev, x)
            new_carry.append(h_t)

        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
        return tuple(new_carry), logits

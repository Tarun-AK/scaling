"""Metric helpers."""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def _prepend_bos(tokens: jax.Array, bos_token_id: int) -> jax.Array:
    bos = jnp.full((tokens.shape[0], 1), bos_token_id, dtype=tokens.dtype)
    return jnp.concatenate([bos, tokens[:, :-1]], axis=1)


def compute_all_ngram_losses(
    apply_fn,
    params,
    tokens: jax.Array,
    bos_token_id: int,
) -> Dict[str, jax.Array]:
    """Compute loss at every position n from 1 to seq_len.

    Uses a single forward pass. The n-gram loss at position n is defined as
    -E[log p_theta(x_n | x_{<n})], where x_1 is conditioned on a fixed BOS token.

    Args:
        apply_fn: Flax apply function (typically `state.apply_fn`).
        params: Model parameters pytree (typically `state.params`).
        tokens: Integer token ids, shape (batch, seq_len).
        bos_token_id: Token id to use as fixed BOS.
    Returns:
        Dict mapping "ngram_1".."ngram_{seq_len}" to scalar losses.
    """
    inputs = _prepend_bos(tokens, bos_token_id)  # (batch, seq_len)
    targets = tokens  # (batch, seq_len)

    logits = apply_fn({"params": params}, inputs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_logp = jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(
        -1
    )

    per_position_loss = -jnp.mean(token_logp, axis=0)

    return {
        f"ngram_{n}": per_position_loss[n - 1] for n in range(1, tokens.shape[1] + 1)
    }

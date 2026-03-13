"""Metric helpers."""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def compute_all_ngram_losses(
    apply_fn, params, tokens: jax.Array
) -> Dict[str, jax.Array]:
    """Compute loss at every position n from 1 to seq_len-1.

    Uses a single forward pass. The n-gram loss at position n is defined as
    -E[log p_theta(x_{n+1} | x_{1:n})], i.e. the cross entropy at position n
    conditioning on all previous tokens.

    Args:
        apply_fn: Flax apply function (typically `state.apply_fn`).
        params: Model parameters pytree (typically `state.params`).
        tokens: Integer token ids, shape (batch, seq_len).

    Returns:
        Dict mapping "ngram_1".."ngram_{seq_len-1}" to scalar losses.
    """
    inputs = tokens[:, :-1]                                          # (batch, seq_len-1)
    targets = tokens[:, 1:]                                          # (batch, seq_len-1)

    logits = apply_fn({"params": params}, inputs)                    # (batch, seq_len-1, vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=-1)                  # (batch, seq_len-1, vocab_size)
    token_logp = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)                                                     # (batch, seq_len-1)

    per_position_loss = -jnp.mean(token_logp, axis=0)               # (seq_len-1,)

    return {f"ngram_{n}": per_position_loss[n - 1] for n in range(1, tokens.shape[1])}

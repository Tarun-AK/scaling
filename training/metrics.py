"""Metric helpers."""

from __future__ import annotations

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
) -> jax.Array:
    """Compute loss at every position n from 1 to seq_len.

    Uses a single forward pass. The n-gram loss at position n is defined as
    -E[log p_theta(x_n | x_{<n})], where x_1 is conditioned on a fixed BOS token.

    Args:
        apply_fn: Flax apply function (typically `state.apply_fn`).
        params: Model parameters pytree (typically `state.params`).
        tokens: Integer token ids, shape (batch, seq_len).
        bos_token_id: Token id to use as fixed BOS.
    Returns:
        Array of shape (seq_len,) -- index n-1 is the ngram_n loss. Returned
        as a single array (not a dict of seq_len python-level scalars) so
        callers can pull results off device with one jax.device_get instead
        of one blocking device_get per sequence position.
    """
    inputs = _prepend_bos(tokens, bos_token_id)  # (batch, seq_len)
    targets = tokens  # (batch, seq_len)

    logits = apply_fn({"params": params}, inputs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_logp = jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(
        -1
    )

    return -jnp.mean(token_logp, axis=0)

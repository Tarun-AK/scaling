"""Loss functions.
In autoregressive language modeling we model p(x) via next-token conditionals.
Even though this is often implemented with a (logits, targets) cross-entropy, the
"targets" are simply the observed tokens from the dataset (shifted by one).
All functions in this module are JAX-friendly and can be used under `jax.jit`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mean_neg_log_prob(logits: jax.Array, observed_tokens: jax.Array) -> jax.Array:
    """Compute mean negative log-probability of observed tokens under logits.

    Args:
        logits: Unnormalized logits, shape (batch, seq_len, vocab_size).
        observed_tokens: Observed token ids aligned to logits, shape (batch, seq_len).
            For an LM this is typically the input sequence shifted by one.

    Returns:
        Scalar mean negative log-probability.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_logp = jnp.take_along_axis(
        log_probs, observed_tokens[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.mean(token_logp)


def cross_entropy_loss(logits: jax.Array, observed_tokens: jax.Array) -> jax.Array:
    """Alias for mean negative log-probability.

    Kept for backwards readability in the trainer.
    """
    return mean_neg_log_prob(logits, observed_tokens)

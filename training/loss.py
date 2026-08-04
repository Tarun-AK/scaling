"""Loss functions.
In autoregressive language modeling we model p(x) via next-token conditionals.
Even though this is often implemented with a (logits, targets) cross-entropy, the
"targets" are simply the observed tokens from the dataset (shifted by one).
All functions in this module are JAX-friendly and can be used under `jax.jit`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


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
    """Mean negative log-probability, without materializing log-softmax.

    Mathematically identical to mean_neg_log_prob, but the fused form avoids
    building a second (batch, seq_len, vocab_size) array for the log
    probabilities and keeping it live for the backward pass. At seq_len 8192 and
    vocab 10000 that intermediate is gigabytes, so it is worth not having.
    """
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, observed_tokens)
    )

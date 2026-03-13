"""Simple NumPy-to-JAX dataloader.

This module provides a lightweight generator that yields batches of token sequences
as JAX arrays. It is deliberately minimal to keep the project easy to understand.
"""

from __future__ import annotations

from typing import Generator, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np


def batch_iterator(
    data: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[jax.Array]:
    """Yield batches from a token chunk array.

    Args:
        data: NumPy array of shape (num_chunks, seq_len) with int token ids.
        batch_size: Batch size.
        shuffle: Whether to shuffle the order of chunks (recommended for train).
        seed: RNG seed used for shuffling.
        drop_last: If True, drop the final partial batch.

    Yields:
        JAX array of shape (batch_size, seq_len), dtype int32.
    """

    num_examples = int(data.shape[0])
    indices = np.arange(num_examples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    # Iterate by slicing the permuted indices.
    start = 0
    while start < num_examples:
        end = start + batch_size
        if end > num_examples:
            if drop_last:
                break
            batch_idx = indices[start:num_examples]
        else:
            batch_idx = indices[start:end]

        batch_np = data[batch_idx]
        # Convert to JAX array (on device). Keep dtype int32.
        yield jnp.asarray(batch_np, dtype=jnp.int32)
        start = end


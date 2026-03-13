"""Dataset utilities.
This module loads WikiText via HuggingFace `datasets`, tokenizes with a custom
BPE tokenizer trained on WikiText-103 with vocab size 8192 (see data/train_tokenizer.py),
then chunks the token stream into fixed-length sequences suitable for language modeling.
All returned arrays are NumPy arrays of dtype int32 with shape (num_chunks, seq_len).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer

TOKENIZER_PATH = "data/tokenizer/tokenizer.json"


@dataclass(frozen=True)
class TokenizedSplits:
    """Container for tokenized dataset splits."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _load_tokenizer() -> HFTokenizer:
    """Load the custom BPE tokenizer from disk.

    Raises:
        FileNotFoundError: If the tokenizer has not been trained yet.
    """
    try:
        return HFTokenizer.from_file(TOKENIZER_PATH)
    except Exception:
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run `python data/train_tokenizer.py` first."
        )


def _tokenize_texts(tokenizer: HFTokenizer, texts: list[str]) -> np.ndarray:
    """Tokenize a list of strings into a 1D NumPy array of token ids.

    Args:
        tokenizer: A HuggingFace tokenizers Tokenizer.
        texts: List of text strings.

    Returns:
        A 1D NumPy array of token ids.
    """
    encoded = tokenizer.encode_batch(texts)
    flat_ids: list[int] = []
    for e in encoded:
        flat_ids.extend(e.ids)
    return np.asarray(flat_ids, dtype=np.int32)


def _chunk_tokens(token_ids: np.ndarray, *, seq_len: int) -> np.ndarray:
    """Chunk a 1D token array into fixed-length sequences.

    Drops any remainder that doesn't fit into an even number of chunks.

    Args:
        token_ids: 1D array of token ids.
        seq_len: Sequence length per chunk.

    Returns:
        Array of shape (num_chunks, seq_len).
    """
    total_len = int(token_ids.shape[0])
    num_chunks = total_len // seq_len
    if num_chunks == 0:
        return np.zeros((0, seq_len), dtype=np.int32)
    trimmed = token_ids[: num_chunks * seq_len]
    return trimmed.reshape(num_chunks, seq_len)


def load_wikitext_tokenized(
    *,
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    vocab_size: int = 8192,
) -> TokenizedSplits:
    """Load WikiText, tokenize, and chunk into fixed-length sequences.

    Args:
        dataset_name: HF dataset name, e.g. "wikitext".
        dataset_config: HF dataset config, e.g. "wikitext-103-raw-v1".
        seq_len: Fixed sequence length for chunking.
        vocab_size: Expected tokenizer vocabulary size; used as a sanity check.

    Returns:
        TokenizedSplits containing train/validation/test chunk arrays.
    """
    ds = load_dataset(dataset_name, dataset_config)
    tokenizer = _load_tokenizer()

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != vocab_size:
        print(
            f"Warning: tokenizer vocab size {actual_vocab} != config vocab_size {vocab_size}"
        )

    def process_split(split_name: str) -> np.ndarray:
        texts = [t for t in ds[split_name]["text"] if t and t.strip()]
        token_ids = _tokenize_texts(tokenizer, texts)
        return _chunk_tokens(token_ids, seq_len=seq_len)

    return TokenizedSplits(
        train=process_split("train"),
        validation=process_split("validation"),
        test=process_split("test"),
    )


def load_splits_as_arrays(
    *,
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning (train, validation, test) arrays."""
    splits = load_wikitext_tokenized(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )
    return splits.train, splits.validation, splits.test

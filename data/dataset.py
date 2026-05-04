"""Dataset utilities.
This module loads WikiText via HuggingFace `datasets`, tokenizes with a custom
BPE tokenizer trained on WikiText-103 with vocab size 8192 (see data/train_tokenizer.py),
then chunks the token stream into fixed-length sequences suitable for language modeling.
All returned arrays are NumPy arrays of dtype int32 with shape (num_chunks, seq_len).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer
from tqdm import tqdm

from data.openwebtext import load_openwebtext_dataset
from data.pg19 import load_pg19_dataset

TOKENIZER_PATH_DEFAULT = "data/tokenizer/tokenizer.json"
CACHE_DIR_DEFAULT = "data/cache"
TOKENIZE_BATCH_SIZE_DEFAULT = 32


@dataclass(frozen=True)
class TokenizedSplits:
    """Container for tokenized dataset splits."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _load_tokenizer(tokenizer_path: str = TOKENIZER_PATH_DEFAULT) -> HFTokenizer:
    """Load the custom BPE tokenizer from disk.

    Raises:
        FileNotFoundError: If the tokenizer has not been trained yet.
    """
    try:
        return HFTokenizer.from_file(tokenizer_path)
    except Exception:
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run `python data/train_tokenizer.py` first."
        )


def _tokenizer_hash(tokenizer_path: str = TOKENIZER_PATH_DEFAULT) -> str:
    data = Path(tokenizer_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _cache_base_name(
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    vocab_size: int,
    tokenizer_hash: str,
) -> str:
    safe_name = dataset_name.replace("/", "_")
    normalized_config = _normalize_dataset_config(dataset_config)
    safe_config = (normalized_config or "none").replace("/", "_")
    return (
        f"{safe_name}_{safe_config}_seq{seq_len}_vocab{vocab_size}_"
        f"tok{tokenizer_hash[:8]}"
    )


def _normalize_dataset_config(dataset_config: str | None) -> str | None:
    if dataset_config is None:
        return None
    value = str(dataset_config).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _cache_paths(
    *,
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    vocab_size: int,
    cache_dir: str,
    tokenizer_path: str,
) -> dict:
    tokenizer_hash = _tokenizer_hash(tokenizer_path)
    base = _cache_base_name(
        dataset_name, dataset_config, seq_len, vocab_size, tokenizer_hash
    )
    root = Path(cache_dir)
    return {
        "train": root / f"{base}_train.npy",
        "validation": root / f"{base}_validation.npy",
        "test": root / f"{base}_test.npy",
        "meta": root / f"{base}_meta.json",
    }


def _load_cached_splits(paths: dict) -> TokenizedSplits:
    missing = [k for k in ("train", "validation", "test") if not paths[k].exists()]
    if missing:
        raise FileNotFoundError("Missing cached arrays: " + ", ".join(missing))
    return TokenizedSplits(
        train=np.load(paths["train"], mmap_mode="r"),
        validation=np.load(paths["validation"], mmap_mode="r"),
        test=np.load(paths["test"], mmap_mode="r"),
    )


def _save_cached_splits(
    paths: dict,
    splits: TokenizedSplits,
    *,
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    vocab_size: int,
    tokenizer_path: str,
) -> None:
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["train"], splits.train)
    np.save(paths["validation"], splits.validation)
    np.save(paths["test"], splits.test)
    meta = {
        "dataset_name": dataset_name,
        "dataset_config": _normalize_dataset_config(dataset_config),
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tokenizer_path": tokenizer_path,
        "tokenizer_hash": _tokenizer_hash(tokenizer_path),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))


def _tokenize_texts(
    tokenizer: HFTokenizer,
    texts: list[str],
    *,
    batch_size: int = 1024,
    desc: str | None = None,
) -> np.ndarray:
    """Tokenize a list of strings into a 1D NumPy array of token ids.

    Args:
        tokenizer: A HuggingFace tokenizers Tokenizer.
        texts: List of text strings.

    Returns:
        A 1D NumPy array of token ids.
    """
    flat_ids: list[int] = []
    total = len(texts)
    batch_indices = range(0, total, batch_size)
    if desc:
        batch_indices = tqdm(batch_indices, desc=desc)
    for start in batch_indices:
        batch = texts[start : start + batch_size]
        encoded = tokenizer.encode_batch(batch)
        for e in encoded:
            flat_ids.extend(e.ids)
    return np.asarray(flat_ids, dtype=np.int32)


def _batch_texts(
    raw_texts: Iterable[str],
    *,
    batch_size: int,
    desc: str,
) -> Iterator[list[str]]:
    total = len(raw_texts) if hasattr(raw_texts, "__len__") else None
    batch: list[str] = []
    for t in tqdm(raw_texts, desc=desc, total=total):
        if not t or not t.strip():
            continue
        batch.append(t)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _count_tokens(
    tokenizer: HFTokenizer,
    raw_texts: Iterable[str],
    *,
    batch_size: int,
    desc: str,
) -> int:
    total_tokens = 0
    for batch in _batch_texts(raw_texts, batch_size=batch_size, desc=desc):
        encoded = tokenizer.encode_batch(batch)
        total_tokens += sum(len(e.ids) for e in encoded)
    return total_tokens


def _tokenize_to_memmap(
    tokenizer: HFTokenizer,
    raw_texts: Iterable[str],
    *,
    seq_len: int,
    out_path: Path,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    total_tokens = _count_tokens(
        tokenizer,
        raw_texts,
        batch_size=batch_size,
        desc=f"Counting {desc}",
    )
    num_chunks = total_tokens // seq_len
    out_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.int32,
        shape=(num_chunks, seq_len),
    )
    buffer: list[int] = []
    start = 0
    idx = 0
    for batch in _batch_texts(
        raw_texts, batch_size=batch_size, desc=f"Tokenizing {desc}"
    ):
        encoded = tokenizer.encode_batch(batch)
        for e in encoded:
            buffer.extend(e.ids)
        while (len(buffer) - start) >= seq_len and idx < num_chunks:
            memmap[idx] = buffer[start : start + seq_len]
            start += seq_len
            idx += 1
        if start > 100_000:
            buffer = buffer[start:]
            start = 0
    memmap.flush()
    return memmap


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
    dataset_config: str | None,
    seq_len: int,
    vocab_size: int = 8192,
    cache_dir: str = CACHE_DIR_DEFAULT,
    require_cache: bool = True,
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE_DEFAULT,
    tokenizer_path: str = TOKENIZER_PATH_DEFAULT,
    dataset_path: str | None = None,
) -> TokenizedSplits:
    """Load WikiText, tokenize, and chunk into fixed-length sequences.

    Args:
        dataset_name: HF dataset name, e.g. "wikitext".
        dataset_config: HF dataset config, e.g. "wikitext-103-raw-v1".
        seq_len: Fixed sequence length for chunking.
        vocab_size: Expected tokenizer vocabulary size; used as a sanity check.
        cache_dir: Directory containing cached token arrays.
        require_cache: If True, fail when cache is missing.

    Returns:
        TokenizedSplits containing train/validation/test chunk arrays.
    """
    paths = _cache_paths(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        tokenizer_path=tokenizer_path,
    )
    try:
        return _load_cached_splits(paths)
    except FileNotFoundError:
        if require_cache:
            raise FileNotFoundError(
                "Cached tokenized dataset not found. Run "
                "`python data/preprocess_wikitext.py` first."
            )

    return build_wikitext_cache(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        tokenize_batch_size=tokenize_batch_size,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
    )


def build_wikitext_cache(
    *,
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    vocab_size: int = 8192,
    cache_dir: str = CACHE_DIR_DEFAULT,
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE_DEFAULT,
    tokenizer_path: str = TOKENIZER_PATH_DEFAULT,
    dataset_path: str | None = None,
) -> TokenizedSplits:
    """Build and save tokenized cache for WikiText."""
    ds = _load_dataset(dataset_name, dataset_config, cache_dir, dataset_path)
    tokenizer = _load_tokenizer(tokenizer_path)

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != vocab_size:
        print(
            f"Warning: tokenizer vocab size {actual_vocab} != config vocab_size {vocab_size}"
        )

    paths = _cache_paths(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        tokenizer_path=tokenizer_path,
    )
    split_arrays = {}
    for split_name in ("train", "validation", "test"):
        split = ds[split_name]
        raw_texts = split.text if hasattr(split, "text") else split["text"]
        split_arrays[split_name] = _tokenize_to_memmap(
            tokenizer,
            raw_texts,
            seq_len=seq_len,
            out_path=paths[split_name],
            batch_size=tokenize_batch_size,
            desc=split_name,
        )
    meta = {
        "dataset_name": dataset_name,
        "dataset_config": _normalize_dataset_config(dataset_config),
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tokenizer_path": tokenizer_path,
        "tokenizer_hash": _tokenizer_hash(tokenizer_path),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    return TokenizedSplits(
        train=split_arrays["train"],
        validation=split_arrays["validation"],
        test=split_arrays["test"],
    )


def load_splits_as_arrays(
    *,
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    vocab_size: int,
    cache_dir: str = CACHE_DIR_DEFAULT,
    require_cache: bool = True,
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE_DEFAULT,
    tokenizer_path: str = TOKENIZER_PATH_DEFAULT,
    dataset_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning (train, validation, test) arrays."""
    splits = load_wikitext_tokenized(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        require_cache=require_cache,
        tokenize_batch_size=tokenize_batch_size,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
    )
    return splits.train, splits.validation, splits.test


def _load_dataset(
    dataset_name: str,
    dataset_config: str | None,
    cache_dir: str,
    dataset_path: str | None,
):
    normalized_config = _normalize_dataset_config(dataset_config)
    if dataset_name in {"deepmind/pg19", "pg19"}:
        return load_pg19_dataset(cache_dir=cache_dir)
    if dataset_name in {"openwebtext", "owt"}:
        return load_openwebtext_dataset(data_dir=dataset_path or "openwebtext")
    if normalized_config:
        return load_dataset(dataset_name, normalized_config)
    return load_dataset(dataset_name)

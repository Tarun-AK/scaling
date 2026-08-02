from __future__ import annotations


def ngram_prefixes_for_split(split: str) -> list[str]:
    """Return metric prefixes to scan for a requested n-gram split."""

    split_norm = split.strip().lower()
    split_map = {
        "combined": ["combined/ngram_", "combined/n_gram_"],
        "validation": [
            "val/ngram_",
            "validation/ngram_",
            "val/n_gram_",
            "validation/n_gram_",
        ],
        "val": [
            "val/ngram_",
            "validation/ngram_",
            "val/n_gram_",
            "validation/n_gram_",
        ],
        "train": ["train/ngram_", "train/n_gram_", "train_ngram/ngram_", "train_ngram/n_gram_"],
        "test": ["test/ngram_", "test/n_gram_"],
        "all": [
            "combined/ngram_",
            "combined/n_gram_",
            "train/ngram_",
            "train/n_gram_",
            "test/ngram_",
            "test/n_gram_",
            "val/ngram_",
            "validation/ngram_",
            "val/n_gram_",
            "validation/n_gram_",
            "train_ngram/ngram_",
            "train_ngram/n_gram_",
        ],
    }
    if split_norm not in split_map:
        raise RuntimeError(
            "Unsupported split. Expected one of ['combined', 'validation', 'val', 'train', 'test', 'all']"
        )
    prefixes = split_map[split_norm]
    return list(dict.fromkeys(prefixes))

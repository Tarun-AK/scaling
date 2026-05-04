"""Preprocess WikiText and cache tokenized arrays.

Run once before training:
    python data/preprocess_wikitext.py
"""

from __future__ import annotations

import argparse

from data.dataset import CACHE_DIR_DEFAULT, TOKENIZER_PATH_DEFAULT, build_wikitext_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--tokenize-batch-size", type=int, default=32)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH_DEFAULT)
    parser.add_argument("--dataset-path", default=None)
    args = parser.parse_args()

    build_wikitext_cache(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        cache_dir=args.cache_dir,
        tokenize_batch_size=args.tokenize_batch_size,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
    )
    print(f"Cached tokenized arrays in {args.cache_dir}")


if __name__ == "__main__":
    main()

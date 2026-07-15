"""Create a local OpenWebText subsample.

This follows the Stanford CS336 recipe, but defaults to a 5M/120k split.

It writes two files that the existing loader already understands:
- ``owt_train.txt``
- ``owt_valid.txt``

Each document is separated by ``<|endoftext|>``.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_TRAIN_SIZE = 5_000_000
DEFAULT_VALID_SIZE = 120_000
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = "openwebtext_5m"


def _write_split(output_path: Path, rows) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(output_path, "w", encoding="utf-8") as handle:
        buffer: list[str] = []
        for row in tqdm(rows, desc=output_path.name):
            buffer.append(row["text"] + "<|endoftext|>")
            if len(buffer) > 1000:
                handle.write("".join(buffer))
                buffer = []
        if buffer:
            handle.write("".join(buffer))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write owt_train.txt and owt_valid.txt",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help="Number of documents to put in the training split",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=DEFAULT_VALID_SIZE,
        help="Number of documents to put in the validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the split",
    )
    args = parser.parse_args()

    dataset = load_dataset("Skylion007/openwebtext")["train"]
    split_dataset = dataset.train_test_split(
        train_size=args.train_size,
        test_size=args.valid_size,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    _write_split(output_dir / "owt_train.txt", split_dataset["train"])
    _write_split(output_dir / "owt_valid.txt", split_dataset["test"])

    print(f"Wrote OpenWebText subset to {output_dir}")
    print(f"train_size={args.train_size} valid_size={args.valid_size} seed={args.seed}")


if __name__ == "__main__":
    main()

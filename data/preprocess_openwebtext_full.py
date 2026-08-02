"""Build a full-OpenWebText corpus with its own train/validation/test split.

Pulls all ~8.01M OpenWebText documents (~10.3B tokens at vocab 10000) and
partitions them into three disjoint splits, written in the same
``<|endoftext|>``-joined format the existing loader understands.

This is a self-contained corpus: it does not reference, reuse, or exclude
anything from the CS336 subset in ``openwebtext/``.

Assignment is by document digest, so the partition is deterministic and
streaming -- no shuffle buffer and no second pass over 38GB of text. The same
document always lands in the same split, so the corpus can be rebuilt or
extended without reshuffling what came before.

Usage:
    python data/preprocess_openwebtext_full.py --output-dir openwebtext_full
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from data.openwebtext import END_OF_TEXT_TOKEN


DEFAULT_DATASET = "Skylion007/openwebtext"
DEFAULT_OUTPUT_DIR = "openwebtext_full"

# Per-mille of documents held out for each eval split. 7/1000 of 8.01M is ~56k
# documents, ~72M tokens -- the same order as the training corpus's own eval
# splits, and large enough for stable held-out loss.
DEFAULT_VALID_PER_MILLE = 7
DEFAULT_TEST_PER_MILLE = 7

WRITE_BUFFER_DOCS = 1000


def _bucket(text: str) -> int:
    """Deterministic per-mille bucket for a document."""
    digest = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest[:4], "big") % 1000


class _DocWriter:
    """Buffered ``<|endoftext|>``-joined document writer."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = io.open(path, "w", encoding="utf-8")
        self._buffer: list[str] = []
        self.count = 0
        self.chars = 0

    def write(self, text: str) -> None:
        self._buffer.append(text + END_OF_TEXT_TOKEN)
        self.count += 1
        self.chars += len(text)
        if len(self._buffer) >= WRITE_BUFFER_DOCS:
            self._flush()

    def _flush(self) -> None:
        if self._buffer:
            self._handle.write("".join(self._buffer))
            self._buffer = []

    def close(self) -> None:
        self._flush()
        self._handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--valid-per-mille",
        type=int,
        default=DEFAULT_VALID_PER_MILLE,
        help="Per-mille of documents held out for validation",
    )
    parser.add_argument(
        "--test-per-mille",
        type=int,
        default=DEFAULT_TEST_PER_MILLE,
        help="Per-mille of documents held out for test",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many source documents (0 = all). For smoke tests.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Stream from the hub instead of materializing the HF arrow cache "
            "(~54GB). Slower, but avoids the local copy."
        ),
    )
    args = parser.parse_args()

    valid_cut = int(args.valid_per_mille)
    test_cut = valid_cut + int(args.test_per_mille)
    if valid_cut < 0 or int(args.test_per_mille) < 0 or test_cut >= 1000:
        raise RuntimeError(
            "--valid-per-mille and --test-per-mille must be >= 0 and sum to < 1000"
        )

    output_dir = Path(args.output_dir)
    dataset = load_dataset(args.dataset, split="train", streaming=bool(args.streaming))

    writers = {
        "train": _DocWriter(output_dir / "owt_train.txt"),
        "validation": _DocWriter(output_dir / "owt_valid.txt"),
        "test": _DocWriter(output_dir / "owt_test.txt"),
    }

    seen = 0
    try:
        for row in tqdm(dataset, desc="partitioning", unit="doc"):
            text = row["text"]
            if not text or not text.strip():
                continue
            seen += 1
            bucket = _bucket(text)
            if bucket < valid_cut:
                split = "validation"
            elif bucket < test_cut:
                split = "test"
            else:
                split = "train"
            writers[split].write(text)
            if args.limit and seen >= args.limit:
                break
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        "dataset": args.dataset,
        "documents_seen": seen,
        "valid_per_mille": valid_cut,
        "test_per_mille": int(args.test_per_mille),
        "limit": args.limit,
        **{
            f"{name}_documents": writer.count for name, writer in writers.items()
        },
        **{f"{name}_chars": writer.chars for name, writer in writers.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    for name in ("train", "validation", "test"):
        if writers[name].count == 0:
            raise RuntimeError(f"Split '{name}' is empty; check the per-mille settings")

    if args.limit:
        print("\n--limit was set: this is a smoke-test corpus, not the full one.")
        return

    print(f"\nWrote OpenWebText corpus to {output_dir}")


if __name__ == "__main__":
    main()

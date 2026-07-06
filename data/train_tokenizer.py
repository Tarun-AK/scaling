"""Train a BPE tokenizer on PG-19.

Run once before training:
    python data/train_tokenizer.py

Saves the tokenizer to data/tokenizer/ which dataset.py will load from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from data.pg19 import load_pg19_dataset


def train_tokenizer(
    save_path: str = "pg19/tokenizer",
    *,
    vocab_size: int = 10000,
    end_of_text_token: str = "<|endoftext|>",
) -> None:
    print("Loading PG-19...")
    ds = load_pg19_dataset()

    # Use only train split for fitting the tokenizer
    texts = []
    for t in tqdm(ds["train"].text, desc="Collecting texts"):
        if t and t.strip():
            texts.append(t)

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=[end_of_text_token],
    )

    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(
        tqdm(texts, desc="Training BPE"),
        trainer=trainer,
    )

    # Insert EOT between documents (OpenWebText-style special token).
    eos_id = tokenizer.token_to_id(end_of_text_token)
    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {end_of_text_token}",
        special_tokens=[(end_of_text_token, eos_id)],
    )

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="pg19/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--eot-token", type=str, default="<|endoftext|>")
    args = parser.parse_args()
    train_tokenizer(
        save_path=args.save_path,
        vocab_size=args.vocab_size,
        end_of_text_token=args.eot_token,
    )

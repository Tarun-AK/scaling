"""Train a BPE tokenizer on WikiText-103 with vocab size 8192.

Run once before training:
    python data/train_tokenizer.py

Saves the tokenizer to data/tokenizer/ which dataset.py will load from.
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from data.pg19 import load_pg19_dataset


def train_tokenizer(save_path: str = "data/tokenizer") -> None:
    print("Loading PG-19...")
    ds = load_pg19_dataset()

    # Use only train split for fitting the tokenizer
    texts = []
    for t in tqdm(ds["train"].text, desc="Collecting texts"):
        if t and t.strip():
            texts.append(t)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=8192,
        special_tokens=["<unk>", "<eos>"],
    )

    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(
        tqdm(texts, desc="Training BPE"),
        trainer=trainer,
    )

    # Insert EOS between documents as the paper describes
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single="$A <eos>",
        special_tokens=[("<eos>", eos_id)],
    )

    tokenizer.save(f"{save_path}/tokenizer.json")
    print(f"Saved tokenizer to {save_path}/tokenizer.json")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    train_tokenizer()

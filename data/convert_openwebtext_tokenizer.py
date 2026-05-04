from __future__ import annotations

import argparse
import pickle
import sys
import zipfile
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(v) for v in cs]))


def _token_bytes_to_string(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token)


def _load_openwebtext_tokenizer(pt_path: Path):
    sys.path.insert(0, str(pt_path.parent))
    import tokenization

    if tokenization is None:
        raise RuntimeError("Failed to import tokenization module")

    with zipfile.ZipFile(pt_path) as archive:
        data_member = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")),
            None,
        )
        if data_member is None:
            raise FileNotFoundError(f"No data.pkl found in {pt_path}")
        return pickle.loads(archive.read(data_member))


def _build_hf_tokenizer(source_tokenizer) -> Tokenizer:
    byte_encoder = _bytes_to_unicode()

    vocab = {
        _token_bytes_to_string(token, byte_encoder): token_id
        for token_id, token in source_tokenizer.id2token.items()
    }
    merges = [
        (
            _token_bytes_to_string(pair[0], byte_encoder),
            _token_bytes_to_string(pair[1], byte_encoder),
        )
        for pair, _ in sorted(
            source_tokenizer.merges.items(),
            key=lambda item: item[1],
        )
    ]

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = getattr(source_tokenizer, "special_tokens", None) or []
    if special_tokens:
        tokenizer.add_special_tokens(
            [
                AddedToken(token, normalized=False, special=True)
                for token in special_tokens
            ]
        )

    return tokenizer


def _validate_equivalence(source_tokenizer, hf_tokenizer: Tokenizer) -> None:
    samples = [
        "Hello world<|endoftext|>",
        " This is a test.",
        "café",
        "line1\nline2",
        "Ωmega",
    ]
    for sample in samples:
        source_ids = source_tokenizer.encode(sample)
        hf_ids = hf_tokenizer.encode(sample).ids
        if source_ids != hf_ids:
            raise RuntimeError(f"Tokenizer mismatch for sample: {sample}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-pt",
        default="openwebtext/tokenizer/owt_tokenizer.pt",
    )
    parser.add_argument(
        "--output-json",
        default="openwebtext/tokenizer/tokenizer.json",
    )
    args = parser.parse_args()

    input_pt = Path(args.input_pt)
    output_json = Path(args.output_json)

    source_tokenizer = _load_openwebtext_tokenizer(input_pt)
    hf_tokenizer = _build_hf_tokenizer(source_tokenizer)
    _validate_equivalence(source_tokenizer, hf_tokenizer)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save(str(output_json))

    print(f"Saved tokenizer json to {output_json}")
    print(f"Vocab size: {hf_tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()

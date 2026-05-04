from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


END_OF_TEXT_TOKEN = "<|endoftext|>"


class OpenWebTextDocuments:
    def __init__(
        self,
        file_path: str | Path,
        *,
        delimiter: str = END_OF_TEXT_TOKEN,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> None:
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[str]:
        buffer = ""
        with self.file_path.open("r", encoding="utf-8") as handle:
            while True:
                chunk = handle.read(self.chunk_size)
                if not chunk:
                    break
                buffer += chunk
                parts = buffer.split(self.delimiter)
                for doc in parts[:-1]:
                    if doc and doc.strip():
                        yield doc
                buffer = parts[-1]
        if buffer and buffer.strip():
            yield buffer


@dataclass(frozen=True)
class OpenWebTextSplit:
    text: OpenWebTextDocuments


def load_openwebtext_dataset(
    data_dir: str = "openwebtext",
) -> dict[str, OpenWebTextSplit]:
    root = Path(data_dir)
    train_path = root / "owt_train.txt"
    valid_path = root / "owt_valid.txt"

    missing = [
        str(path)
        for path in (train_path, valid_path)
        if not path.exists() or not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing OpenWebText files: " + ", ".join(missing))

    return {
        "train": OpenWebTextSplit(text=OpenWebTextDocuments(train_path)),
        "validation": OpenWebTextSplit(text=OpenWebTextDocuments(valid_path)),
        "test": OpenWebTextSplit(text=OpenWebTextDocuments(valid_path)),
    }

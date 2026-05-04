"""PG-19 dataset loader without HuggingFace scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm


PG19_REPO_BASE = "https://huggingface.co/datasets/deepmind/pg19/resolve/main"
PG19_DATA_BASE = "https://storage.googleapis.com/deepmind-gutenberg"
PG19_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Pg19Split:
    text: list[str]


def load_pg19_dataset(cache_dir: str = "data/pg19_cache") -> dict[str, Pg19Split]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    split_files = {
        split: _download_text(
            f"{PG19_REPO_BASE}/data/{split}_files.txt",
            cache_root / f"{split}_files.txt",
        ).splitlines()
        for split in PG19_SPLITS
    }

    data = {}
    for split, files in split_files.items():
        texts = []
        for rel_path in tqdm(files, desc=f"Downloading {split}"):
            rel_path = rel_path.strip()
            if not rel_path:
                continue
            local_path = cache_root / rel_path
            url = f"{PG19_DATA_BASE}/{rel_path}"
            text = _download_text(url, local_path)
            if text and text.strip():
                texts.append(text)
        data[split] = Pg19Split(text=texts)
    return data


def _download_text(url: str, local_path: Path) -> str:
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        content = response.read().decode("utf-8")
    local_path.write_text(content, encoding="utf-8")
    return content

"""Estimate n-gram losses for a W&B group and write them to W&B.

The script loads each finished run in a W&B group, restores its checkpoint,
reloads the tokenized dataset using the run's configured `seq_len`, computes
per-position n-gram losses for the requested split(s), and writes the metrics
back to the originating run in W&B.

Usage:
    python analysis/compute/estimate_ngrams.py --group <group> --split all
    python analysis/compute/estimate_ngrams.py --group <group> --split train
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import wandb
from data.dataloader import batch_iterator
from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from models.mamba import MambaLanguageModel
from models.transformer import TransformerLanguageModel
from models.vanilla_rnn import VanillaRNNLanguageModel
from training.trainer import TrainState, create_train_state, eval_step

DEFAULT_ENTITY = "tarunadvaith-"
DEFAULT_PROJECT = "scaling"
DEFAULT_BATCH_SIZE = 128

SPLIT_ALIASES = {
    "val": "validation",
    "validation": "validation",
    "train": "train",
    "test": "test",
}


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _config_to_namespace(
    config: dict[str, Any], *, batch_size: int | None = None
) -> Namespace:
    values = dict(config)
    if batch_size is not None:
        values["batch_size"] = batch_size
    return Namespace(**values)


def _download_checkpoint_artifact(
    *, entity: str, project: str, run_id: str, api: wandb.Api
) -> str:
    from analysis.plot_bipartite_mi import resolve_final_checkpoint_artifact

    # Resolves checkpoint-<run_id>:latest, falling back to the furthest-trained
    # labelled checkpoint when that redundant artifact has been pruned.
    artifact_name = resolve_final_checkpoint_artifact(run_id, api)
    artifact = api.artifact(artifact_name)
    tmpdir = tempfile.mkdtemp(prefix="ckpt_artifact_")
    artifact_dir = artifact.download(root=tmpdir)
    return os.path.join(artifact_dir, "ckpt")


def _normalize_lstm_params(params: dict[str, Any], num_layers: int) -> dict[str, Any]:
    if any(key.startswith("rnn_") for key in params):
        return params

    normalized = dict(params)
    for layer_idx in range(num_layers):
        cell_key = f"LSTMCell_{layer_idx}"
        if cell_key not in params:
            raise RuntimeError(f"Missing {cell_key} in checkpoint params")
        normalized.setdefault(f"rnn_{layer_idx}", {"cell": params[cell_key]})
    return normalized


def _load_checkpoint(
    ckpt_path: str, state: TrainState, config: dict[str, Any]
) -> TrainState:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    params = restored["params"]
    num_layers = int(config["num_layers"])
    if any(key.startswith("LSTMCell_") for key in params):
        params = _normalize_lstm_params(params, num_layers)
    return state.replace(params=params)


def _load_model(config: dict[str, Any]):
    model_type = str(config.get("model_type", "lstm")).lower()
    hidden_dim = int(config["hidden_dim"])
    num_layers = int(config["num_layers"])
    vocab_size = int(config["vocab_size"])

    if model_type == "lstm":
        return LSTMLanguageModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
        )
    if model_type == "mamba":
        return MambaLanguageModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            state_size=int(
                config.get("mamba_state_size", config.get("mamba_d_state", 16))
            ),
            head_dim=int(config.get("mamba_head_dim", 16)),
            chunk_size=int(config.get("mamba_chunk_size", 256)),
            expand=int(config.get("mamba_expand", 2)),
            conv_kernel=int(config.get("mamba_conv_kernel", 4)),
        )
    if model_type == "vanilla_rnn":
        return VanillaRNNLanguageModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
        )
    if model_type == "transformer":
        return TransformerLanguageModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            num_heads=int(config.get("transformer_num_heads", 8)),
            num_kv_heads=int(
                config.get(
                    "transformer_num_kv_heads", config.get("transformer_num_heads", 8)
                )
            ),
            ffn_mult=float(config.get("transformer_ffn_mult", 8.0 / 3.0)),
            rope_theta=float(config.get("transformer_rope_theta", 1_000_000.0)),
            layer_norm_epsilon=float(
                config.get("transformer_layer_norm_epsilon", 1e-6)
            ),
            tie_word_embeddings=bool(
                config.get("transformer_tie_word_embeddings", True)
            ),
        )
    raise RuntimeError(f"Unsupported model_type '{model_type}'")


def _split_prefix(split: str) -> str:
    if split == "validation":
        return "val"
    return split


def _compute_split_ngram(
    *,
    state: TrainState,
    data: np.ndarray,
    bos_token_id: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    seq_len = int(data.shape[1])
    sums = np.zeros(seq_len, dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.float64)

    with tqdm(total=int(data.shape[0]), desc=desc, unit="seq") as progress:
        for batch in batch_iterator(
            data,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            drop_last=False,
        ):
            batch_size_actual = int(batch.shape[0])
            per_position_loss = eval_step(state, batch, bos_token_id)
            # Single bulk device_get per batch instead of one per sequence
            # position -- avoids seq_len blocking host<->device round trips.
            values = np.asarray(jax.device_get(per_position_loss))
            sums += values * batch_size_actual
            counts += batch_size_actual
            progress.update(batch_size_actual)

    if np.any(counts == 0):
        raise RuntimeError("Encountered empty counts while computing n-gram losses")

    return (sums / counts).astype(np.float32)


def _collect_group_runs(
    api: wandb.Api, entity: str, project: str, group: str
) -> list[wandb.apis.public.Run]:
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    finished = [run for run in runs if run.state == "finished"]
    if not finished:
        raise RuntimeError(f"No finished runs found for group '{group}'")
    return finished


def _validate_group_seq_len(runs: list[wandb.apis.public.Run]) -> int:
    seq_lens = {
        int((run.config or {}).get("seq_len"))
        for run in runs
        if (run.config or {}).get("seq_len") is not None
    }
    if not seq_lens:
        raise RuntimeError("No seq_len found in group runs")
    if len(seq_lens) != 1:
        raise RuntimeError(
            f"Expected a single seq_len for the group, found {sorted(seq_lens)}"
        )
    return next(iter(seq_lens))


def _normalize_splits(split: str) -> list[str]:
    split_norm = split.strip().lower()
    if split_norm == "all":
        return ["train", "validation", "test"]
    if split_norm in SPLIT_ALIASES:
        return [SPLIT_ALIASES[split_norm]]
    raise RuntimeError(
        "Unsupported split. Expected one of: train, test, val, validation, all"
    )


def _wandb_log_split_metrics(
    *,
    entity: str,
    project: str,
    run_id: str,
    split: str,
    ngram: np.ndarray,
) -> None:
    prefix = _split_prefix(split)
    metrics = {
        f"{prefix}/n_gram_{idx + 1}": float(value) for idx, value in enumerate(ngram)
    }
    wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        reinit=True,
    )
    try:
        wandb.run.summary.update(metrics)
    finally:
        wandb.finish()


def _prepare_state(config: dict[str, Any]) -> TrainState:
    model = _load_model(config)
    init_config = _config_to_namespace(config, batch_size=1)
    rng = jax.random.PRNGKey(0)
    return create_train_state(model, init_config, rng)


def _load_split_arrays(
    *,
    config: dict[str, Any],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_name = str(config["dataset_name"])
    dataset_config = config.get("dataset_config")
    cache_dir = _resolve_path(str(config.get("cache_dir", "data/cache"))) or str(
        REPO_ROOT / "data" / "cache"
    )
    tokenizer_path = _resolve_path(
        str(config.get("tokenizer_path", "data/tokenizer/tokenizer.json"))
    )
    dataset_path = _resolve_path(config.get("dataset_path"))

    return load_splits_as_arrays(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=int(config["vocab_size"]),
        cache_dir=cache_dir,
        require_cache=False,
        tokenize_batch_size=int(config.get("tokenize_batch_size", 32)),
        tokenizer_path=tokenizer_path
        or str(REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"),
        dataset_path=dataset_path,
    )


def _run_for_split(
    *,
    api: wandb.Api,
    entity: str,
    project: str,
    run: wandb.apis.public.Run,
    split: str,
    seq_len: int,
) -> None:
    config = dict(run.config or {})
    config["mamba_chunk_size"] = 64
    if "seq_len" in config and int(config["seq_len"]) != seq_len:
        print(
            f"Warning: run {run.id} seq_len={config['seq_len']} does not match group seq_len={seq_len}; using group seq_len"
        )

    split_key = _split_prefix(split)

    state = _prepare_state(config)
    ckpt_path = _download_checkpoint_artifact(
        entity=run.entity,
        project=run.project,
        run_id=run.id,
        api=api,
    )
    state = _load_checkpoint(ckpt_path, state, config)

    train_np, val_np, test_np = _load_split_arrays(config=config, seq_len=seq_len)
    split_arrays = {
        "train": train_np,
        "validation": val_np,
        "test": test_np,
    }
    data = split_arrays[split]

    bos_token_id = int(config.get("bos_token_id", 0))
    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    ngram = _compute_split_ngram(
        state=state,
        data=data,
        bos_token_id=bos_token_id,
        batch_size=batch_size,
        desc=f"{run.id}/{split_key}",
    )
    _wandb_log_split_metrics(
        entity=entity,
        project=project,
        run_id=run.id,
        split=split,
        ngram=ngram,
    )
    print(f"Computed and logged n-grams for run={run.id} split={split}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and cache per-position n-gram losses"
    )
    parser.add_argument("--group", required=True, help="W&B group to process")
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test", "val", "validation", "all"],
        help="Split to process",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="W&B entity")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project")
    args = parser.parse_args()

    api = wandb.Api()
    runs = _collect_group_runs(api, args.entity, args.project, args.group)
    seq_len = _validate_group_seq_len(runs)
    splits = _normalize_splits(args.split)

    print(f"Processing group={args.group} seq_len={seq_len} splits={splits}")
    for run in runs:
        for split in splits:
            _run_for_split(
                api=api,
                entity=args.entity,
                project=args.project,
                run=run,
                split=split,
                seq_len=seq_len,
            )


if __name__ == "__main__":
    main()

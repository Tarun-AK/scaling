"""Recompute and log combined eval metrics for finished runs in a W&B group.

This mirrors the post-training combined evaluation path in `training/trainer.py`,
but restores model weights from W&B checkpoint artifacts (same pattern used by
`analysis/plot_bipartite_mi.py`) instead of local checkpoint folders.

Usage:
    python analysis/combined_evals.py --group seq_len_2048_openwebtext_mamba_sweep
    python analysis/combined_evals.py --group <group> --hidden-dim 80
"""

from __future__ import annotations

import argparse
import re
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

import wandb
from analysis.plot_bipartite_mi import (
    WANDB_PROJECT,
    _download_checkpoint_artifact,
    _filter_runs_by_hidden_dim,
    _load_checkpoint,
    _resolve_group_runs,
)
from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from models.mamba import MambaLanguageModel
from models.vanilla_rnn import VanillaRNNLanguageModel
from data.dataloader import batch_iterator
from training.trainer import _eval_loss, create_train_state, eval_step


def _download_checkpoint_for_run(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    cache_dir: str,
) -> str:
    """Download checkpoint artifact for a run.

    Uses the exact `plot_bipartite_mi` logic first (`checkpoint-<run_id>:latest`).
    If unavailable, falls back to latest per-epoch artifact
    (`checkpoint-<run_id>-epoch-####:v*`) for runs that only logged epoch checkpoints.
    """
    try:
        return _download_checkpoint_artifact(run.id, api, cache_dir)
    except Exception as exc:
        pattern = re.compile(rf"^checkpoint-{re.escape(run.id)}-epoch-(\d{{4}}):v\d+$")
        best_artifact_name = None
        best_epoch = -1
        for artifact in run.logged_artifacts():
            match = pattern.match(artifact.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_artifact_name = artifact.name

        if best_artifact_name is None:
            raise RuntimeError(
                "No checkpoint artifact found for run "
                f"{run.id}. Expected checkpoint-{run.id}:latest or per-epoch artifacts."
            ) from exc

        print(
            "  Falling back to latest epoch checkpoint artifact: "
            f"{best_artifact_name}"
        )
        artifact = api.artifact(f"{WANDB_PROJECT}/{best_artifact_name}")
        artifact_dir = artifact.download(root=f"{cache_dir}/{run.id}")
        return f"{artifact_dir}/ckpt"


def _build_model(cfg: dict[str, Any]):
    model_type = str(cfg.get("model_type", "lstm")).lower()
    if model_type == "lstm":
        return LSTMLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    if model_type == "mamba":
        return MambaLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
            state_size=int(cfg.get("mamba_state_size", cfg.get("mamba_d_state", 16))),
            head_dim=int(cfg.get("mamba_head_dim", 16)),
            chunk_size=int(cfg.get("mamba_chunk_size", 256)),
            expand=int(cfg.get("mamba_expand", 2)),
            conv_kernel=int(cfg.get("mamba_conv_kernel", 4)),
        )
    if model_type == "vanilla_rnn":
        return VanillaRNNLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    raise RuntimeError(
        "Unsupported model_type in run config. "
        f"Expected one of ['lstm', 'mamba', 'vanilla_rnn'], got '{model_type}'"
    )


def _load_split_arrays(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_path = cfg.get("dataset_path")
    return load_splits_as_arrays(
        dataset_name=str(cfg["dataset_name"]),
        dataset_config=cfg.get("dataset_config"),
        seq_len=int(cfg["seq_len"]),
        vocab_size=int(cfg["vocab_size"]),
        cache_dir=str(cfg.get("cache_dir", "data/cache")),
        require_cache=bool(cfg.get("require_cached_data", True)),
        tokenize_batch_size=int(cfg.get("tokenize_batch_size", 32)),
        tokenizer_path=str(cfg.get("tokenizer_path", "data/tokenizer/tokenizer.json")),
        dataset_path=str(dataset_path) if dataset_path is not None else None,
    )


def _state_cfg_from_run_config(cfg: dict[str, Any], max_seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=int(cfg.get("batch_size", 1)),
        seq_len=int(cfg.get("seq_len", max_seq_len)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )


def _eval_split_with_progress(
    *,
    state,
    data: np.ndarray,
    batch_size: int,
    bos_token_id: int,
    label: str,
) -> dict[str, float]:
    batches = list(
        batch_iterator(
            data,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            drop_last=True,
        )
    )
    total = len(batches)
    if total == 0:
        return {}

    print(f"  {label}: evaluating {total} batches")
    totals: dict[str, float] = {}
    count = 0
    for idx, batch in enumerate(batches, start=1):
        batch_metrics = eval_step(state, batch, bos_token_id)
        count += 1
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(jax.device_get(value))
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"  {label}: batch {idx}/{total}")
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, help="W&B group to evaluate")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/combined_evals_cache",
        help="Directory used to cache downloaded checkpoint artifacts",
    )
    parser.add_argument(
        "--log-test",
        action="store_true",
        help="Also log test/loss and test/ngram_* alongside combined/ngram_*",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = _resolve_group_runs(api, args.group)
    runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, args.group)

    entity, project = WANDB_PROJECT.split("/")

    for run_idx, run in enumerate(runs, start=1):
        cfg = dict(run.config or {})
        hidden_dim = int(cfg["hidden_dim"])
        print(
            f"\n[{run_idx}/{len(runs)}] Evaluating hidden_dim={hidden_dim}, "
            f"run={run.id} ({run.name})"
        )

        print("  Loading cached dataset splits...")
        train_np, val_np, test_np = _load_split_arrays(cfg)
        combined_np = np.concatenate([val_np, test_np], axis=0)
        bos_token_id = int(cfg.get("bos_token_id", 0))

        print("  Downloading/restoring checkpoint artifact...")
        ckpt_path = _download_checkpoint_for_run(run, api, args.cache_dir)

        print("  Building model/state...")
        model = _build_model(cfg)
        state = create_train_state(
            model,
            _state_cfg_from_run_config(cfg, max_seq_len=int(cfg["seq_len"])),
            jax.random.PRNGKey(0),
        )

        state, restored = _load_checkpoint(ckpt_path, state)
        ckpt_run_id = restored.get("wandb_run_id")
        if ckpt_run_id is not None and ckpt_run_id != run.id:
            raise RuntimeError(
                "Checkpoint/run mismatch for hidden_dim="
                f"{hidden_dim}: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
            )

        combined_mean = _eval_split_with_progress(
            state=state,
            data=combined_np,
            batch_size=int(cfg["batch_size"]),
            bos_token_id=bos_token_id,
            label="combined",
        )
        if not combined_mean:
            raise RuntimeError(
                f"No combined metrics computed for hidden_dim={hidden_dim} run={run.id}"
            )

        payload = {f"combined/{k}": float(v) for k, v in combined_mean.items()}

        if args.log_test:
            test_loss = _eval_loss(state, test_np, int(cfg["batch_size"]), bos_token_id)
            test_mean = _eval_split_with_progress(
                state=state,
                data=test_np,
                batch_size=int(cfg["batch_size"]),
                bos_token_id=bos_token_id,
                label="test",
            )
            payload["test/loss"] = float(test_loss)
            payload.update({f"test/{k}": float(v) for k, v in test_mean.items()})

        print(
            "  Computed combined metrics: "
            f"{len(combined_mean)} keys, ngram_1={combined_mean.get('ngram_1', float('nan')):.6g}"
        )
        print(f"  Logging to W&B run {run.id}")

        resumed = wandb.init(
            project=project,
            entity=entity,
            id=run.id,
            resume="must",
        )
        wandb.log(payload)
        resumed.finish(exit_code=0)


if __name__ == "__main__":
    main()

"""Compare empirical vs model first-token distribution.

Usage:
  python analysis/sanity_check_first_token.py --group <group> --hidden-dim 128
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import termios
import tty
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb

from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from training.trainer import create_train_state

WANDB_PROJECT = "tarunadvaith-/scaling"
plt.style.use("~/plotStyle.mplstyle")


def _show_image(path: str) -> None:
    if shutil.which("kitten") is None:
        return
    subprocess.run(["kitten", "icat", path], check=False)
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        sys.stdin.read(1)
    subprocess.run(["kitten", "icat", "--clear"], check=False)


def _wandb_entity_project() -> tuple[str, str]:
    entity, project = WANDB_PROJECT.split("/")
    return entity, project


def _download_checkpoint_artifact(run_id: str, api: wandb.Api) -> str:
    entity, project = _wandb_entity_project()
    artifact_name = f"{entity}/{project}/checkpoint-{run_id}:latest"
    artifact = api.artifact(artifact_name)
    tmpdir = tempfile.mkdtemp(prefix="ckpt_artifact_")
    artifact_dir = artifact.download(root=tmpdir)
    return os.path.join(artifact_dir, "ckpt")


def _load_checkpoint(ckpt_path: str, state) -> tuple[Any, dict]:
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return state.replace(params=restored["params"]), restored


def _resolve_run(api: wandb.Api, group: str, hidden_dim: int) -> wandb.apis.public.Run:
    runs = list(
        api.runs(
            WANDB_PROJECT,
            filters={
                "group": group,
                "state": "finished",
                "config.hidden_dim": hidden_dim,
            },
        )
    )
    if not runs:
        raise RuntimeError(
            f"No finished runs found for group='{group}' hidden_dim={hidden_dim}"
        )
    if len(runs) > 1:
        names = ", ".join(r.name for r in runs)
        raise RuntimeError(
            "Multiple finished runs found for "
            f"group='{group}' hidden_dim={hidden_dim}: {names}"
        )
    return runs[0]


def _empirical_first_token_distribution(
    train_np: np.ndarray, vocab_size: int
) -> np.ndarray:
    first_tokens = train_np[:, 0]
    counts = np.bincount(first_tokens, minlength=vocab_size).astype(np.float64)
    total = counts.sum()
    if total == 0:
        raise RuntimeError(
            "Training split is empty; cannot compute empirical distribution."
        )
    return counts / total


def _model_first_token_distribution(
    state, model: LSTMLanguageModel, bos_token_id: int, batch_size: int
) -> np.ndarray:
    inputs = jnp.full((batch_size, 1), bos_token_id, dtype=jnp.int32)
    logits = state.apply_fn({"params": state.params}, inputs)
    probs = jax.nn.softmax(logits, axis=-1)
    mean_probs = jnp.mean(probs, axis=0)[0]
    return np.array(jax.device_get(mean_probs), dtype=np.float64)


def _plot_distributions(
    empirical: np.ndarray,
    model: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    order = np.argsort(-empirical)
    empirical_sorted = empirical[order]
    model_sorted = model[order]

    x = np.arange(len(empirical_sorted))

    plt.figure(figsize=(12, 6))
    plt.step(
        x,
        model_sorted,
        where="mid",
        label="model",
        linewidth=1.0,
        alpha=0.7,
    )
    plt.step(
        x,
        empirical_sorted,
        where="mid",
        label="empirical",
        linewidth=2.5,
        alpha=0.7,
    )
    plt.xlabel("Tokens sorted by empirical frequency (desc)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.yscale("log")
    plt.xlim(0, len(empirical_sorted) - 1)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--output", type=str, default="results/first_token_sanity.png")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    api = wandb.Api()
    run = _resolve_run(api, args.group, args.hidden_dim)
    cfg = run.config or {}

    ckpt_path = _download_checkpoint_artifact(run.id, api)

    model = LSTMLanguageModel(
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        vocab_size=int(cfg["vocab_size"]),
    )
    rng = jax.random.PRNGKey(0)
    state_cfg = SimpleNamespace(
        batch_size=int(cfg["batch_size"]),
        seq_len=int(cfg["seq_len"]),
        learning_rate=float(cfg["learning_rate"]),
    )
    state = create_train_state(model, state_cfg, rng)
    state, restored = _load_checkpoint(ckpt_path, state)
    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    cache_dir = str(cfg.get("cache_dir", "data/cache"))
    require_cache = bool(cfg.get("require_cached_data", True))

    train_np, _, _ = load_splits_as_arrays(
        dataset_name=str(cfg["dataset_name"]),
        dataset_config=str(cfg["dataset_config"]),
        seq_len=int(cfg["seq_len"]),
        vocab_size=int(cfg["vocab_size"]),
        cache_dir=cache_dir,
        require_cache=require_cache,
    )

    empirical = _empirical_first_token_distribution(train_np, int(cfg["vocab_size"]))
    model_dist = _model_first_token_distribution(
        state, model, int(cfg.get("bos_token_id", 0)), args.batch_size
    )

    title = (
        f"First-token distribution (group={args.group}, hidden_dim={args.hidden_dim})"
    )
    _plot_distributions(empirical, model_dist, args.output, title)


if __name__ == "__main__":
    main()

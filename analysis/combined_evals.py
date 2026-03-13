"""Evaluate checkpointed models on concatenated val+test splits (as in the paper),
and on the training set for runs in group "main".

Usage:
    python analysis/eval_combined.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np
import orbax.checkpoint as ocp

import wandb
from data.dataloader import batch_iterator
from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from training.trainer import TrainState, _mean_metrics, create_train_state, eval_step

CHECKPOINT_DIR = "checkpoints"
WANDB_PROJECT = "tarunadvaith-/scaling"


def load_checkpoint(ckpt_path: str, state: TrainState) -> TrainState:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return state.replace(params=restored["params"])


def build_state(cfg: dict, hidden_dim: int, rng: jax.Array) -> TrainState:
    model = LSTMLanguageModel(
        hidden_dim=hidden_dim,
        num_layers=int(cfg["num_layers"]),
        vocab_size=int(cfg["vocab_size"]),
    )

    class _Cfg:
        pass

    state_cfg = _Cfg()
    for k, v in cfg.items():
        setattr(state_cfg, k, v)

    return create_train_state(model, state_cfg, rng)


def eval_split(state: TrainState, data: np.ndarray, batch_size: int) -> dict:
    metrics = [
        eval_step(state, batch)
        for batch in batch_iterator(
            data, batch_size=batch_size, shuffle=False, seed=0, drop_last=True
        )
    ]
    return _mean_metrics(metrics) if metrics else {}


def main() -> None:
    hidden_dim_dirs = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("hidden_dim=")],
        key=lambda x: int(x.split("=")[1]),
    )
    if not hidden_dim_dirs:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return

    api = wandb.Api()
    all_runs = list(api.runs(WANDB_PROJECT))

    for hd_dir in hidden_dim_dirs:
        hidden_dim = int(hd_dir.split("=")[1])
        ckpt_path = os.path.abspath(os.path.join(CHECKPOINT_DIR, hd_dir, "ckpt"))
        print(f"\nEvaluating hidden_dim={hidden_dim}...")

        # Prefer main group run, fall back to any matching run
        matching = [r for r in all_runs if r.config.get("hidden_dim") == hidden_dim]
        main_runs = [r for r in matching if r.group == "main"]
        other_runs = [r for r in matching if r.group != "main"]

        if not matching:
            print(f"  No W&B run found for hidden_dim={hidden_dim}, skipping.")
            continue

        run = main_runs[0] if main_runs else other_runs[0]
        is_main = run.group == "main"
        cfg = run.config

        print(f"  Using run: {run.name} (group={run.group})")

        # Load data
        train_np, val_np, test_np = load_splits_as_arrays(
            dataset_name=cfg["dataset_name"],
            dataset_config=cfg["dataset_config"],
            seq_len=int(cfg["seq_len"]),
            vocab_size=int(cfg["vocab_size"]),
        )
        combined_np = np.concatenate([val_np, test_np], axis=0)

        # Build and load model
        rng = jax.random.PRNGKey(0)
        state = build_state(cfg, hidden_dim, rng)
        state = load_checkpoint(ckpt_path, state)

        wandb.init(
            project=WANDB_PROJECT.split("/")[-1],
            entity=WANDB_PROJECT.split("/")[0],
            id=run.id,
            resume="must",
        )

        # Combined val+test eval
        combined_mean = eval_split(state, combined_np, int(cfg["batch_size"]))
        wandb.log({f"combined/{k}": v for k, v in combined_mean.items()})
        print(f"  Combined done. ngram_1={combined_mean.get('ngram_1', 'N/A'):.4f}")

        # Training n-gram eval only for main group
        if is_main:
            print(f"  Evaluating train n-grams (group=main)...")
            train_mean = eval_split(state, train_np, int(cfg["batch_size"]))
            wandb.log({f"train_ngram/{k}": v for k, v in train_mean.items()})
            print(
                f"  Train n-grams done. ngram_1={train_mean.get('ngram_1', 'N/A'):.4f}"
            )

        wandb.finish()


if __name__ == "__main__":
    main()

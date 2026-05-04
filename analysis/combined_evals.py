"""Evaluate checkpointed models on concatenated val+test splits (as in the paper),
and on the training set for runs in a specified group.

Usage:
    python analysis/combined_evals.py --group main
"""

from __future__ import annotations

import argparse
import os
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

import wandb
from data.dataloader import batch_iterator
from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from training.trainer import TrainState, _mean_metrics, create_train_state, eval_step

WANDB_PROJECT = "tarunadvaith-/scaling"


def load_checkpoint(ckpt_path: str, state: TrainState) -> tuple[TrainState, dict]:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return state.replace(params=restored["params"]), restored


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


def eval_split(
    state: TrainState, data: np.ndarray, batch_size: int, bos_token_id: int
) -> dict:
    batches = list(
        batch_iterator(
            data, batch_size=batch_size, shuffle=False, seed=0, drop_last=True
        )
    )
    metrics = [
        eval_step(state, batch, bos_token_id)
        for batch in tqdm(batches, desc="Batches", leave=False)
    ]
    return _mean_metrics(metrics) if metrics else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="W&B group to evaluate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        action="append",
        default=[],
        dest="hidden_dims",
        help="Specific hidden dim(s) to evaluate (default: all runs in group)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(
        api.runs(WANDB_PROJECT, filters={"group": args.group, "state": "finished"})
    )
    if args.hidden_dims:
        target_hidden_dims = set(args.hidden_dims)
        runs = [
            r for r in runs if int(r.config.get("hidden_dim", -1)) in target_hidden_dims
        ]
    if not runs:
        print(f"No finished runs found in group '{args.group}'")
        return

    runs = sorted(runs, key=lambda r: int(r.config.get("hidden_dim", -1)))

    for run in tqdm(runs, desc="Runs"):
        cfg = run.config
        hidden_dim = int(cfg["hidden_dim"])
        ckpt_path = _download_checkpoint_artifact(run.id, api)
        print(f"\nEvaluating hidden_dim={hidden_dim}, run={run.name}...")
        bos_token_id = int(cfg.get("bos_token_id", 0))

        train_np, val_np, test_np = load_splits_as_arrays(
            dataset_name=cfg["dataset_name"],
            dataset_config=cfg["dataset_config"],
            seq_len=int(cfg["seq_len"]),
            vocab_size=int(cfg["vocab_size"]),
            cache_dir=str(cfg.get("cache_dir", "data/cache")),
            require_cache=bool(cfg.get("require_cached_data", True)),
        )
        combined_np = np.concatenate([val_np, test_np], axis=0)

        rng = jax.random.PRNGKey(0)
        state = build_state(cfg, hidden_dim, rng)
        state, restored = load_checkpoint(ckpt_path, state)
        ckpt_run_id = restored.get("wandb_run_id")
        if ckpt_run_id is None:
            raise RuntimeError(
                f"Checkpoint at {ckpt_path} is missing wandb_run_id; "
                "cannot verify run alignment."
            )
        if ckpt_run_id != run.id:
            raise RuntimeError(
                "Checkpoint/run mismatch for hidden_dim="
                f"{hidden_dim}: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
            )

        combined_mean = eval_split(
            state, combined_np, int(cfg["batch_size"]), bos_token_id
        )
        print(f"  Combined done. ngram_1={combined_mean.get('ngram_1', 'N/A'):.4f}")

        train_mean = eval_split(state, train_np, int(cfg["batch_size"]), bos_token_id)
        print(f"  Train n-grams done. ngram_1={train_mean.get('ngram_1', 'N/A'):.4f}")

        print(f"  Logging to run: {run.name}")
        wandb.init(
            project=WANDB_PROJECT.split("/")[-1],
            entity=WANDB_PROJECT.split("/")[0],
            id=run.id,
            resume="must",
        )
        wandb.log({f"combined/{k}": v for k, v in combined_mean.items()})
        wandb.log({f"train_ngram/{k}": v for k, v in train_mean.items()})
        wandb.finish()


if __name__ == "__main__":
    main()

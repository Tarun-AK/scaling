"""Backfill missing direct/data log q(y) cache values for selected N.

This updates existing direct/data cache entries used by plot_bipartite_mi.py
without forcing full resampling. It computes only missing N values and writes
them into the existing data_log_q_y_*.npz cache for each run.
"""

from __future__ import annotations

import argparse
import os
import re
from types import SimpleNamespace

import jax
import numpy as np

import wandb
from analysis.plot_bipartite_mi import (
    DEFAULT_MAX_N,
    DEFAULT_N_VALUES,
    _build_language_model_from_config,
    _compute_log_q_y_means,
    _data_cache_key,
    _data_cache_paths,
    _load_data_chunks,
    _download_checkpoint_artifact,
    _filter_runs_by_hidden_dim,
    _load_checkpoint,
    _load_data_sample_cache,
    _load_log_q_y_mean_cache,
    _normalize_params_for_step,
    _resolve_group_runs,
    _score_logps,
    _save_data_sample_cache,
    _save_log_q_y_mean_cache,
)
from training.trainer import create_train_state


def _seq_len_required(max_n: int, requested_n_values: list[int]) -> int:
    default_ns = [n for n in DEFAULT_N_VALUES if n <= int(max_n)]
    default_max = int(default_ns[-1]) if default_ns else None
    requested_max = int(max(requested_n_values)) if requested_n_values else None
    if default_max is not None and requested_max is not None:
        return max(default_max, requested_max)
    if default_max is not None:
        return default_max
    if requested_max is not None:
        return requested_max
    raise RuntimeError("No valid N values to determine required seq_len")


def _find_best_seed_log_cache(
    *,
    cache_dir: str,
    run_id: str,
    split: str,
    bos_token_id: int,
    target_seq_len: int,
) -> tuple[str, str] | None:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    if not os.path.isdir(run_cache_dir):
        return None

    pattern = re.compile(
        rf"^data_samples_data_{re.escape(split)}_seq(?P<seq>\d+)_numall_bos{int(bos_token_id)}(?:_seed\d+)?\.npz$"
    )
    candidates: list[tuple[int, str, str]] = []
    for filename in os.listdir(run_cache_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        seq_len = int(match.group("seq"))
        if seq_len >= int(target_seq_len):
            continue
        sample_cache_path = os.path.join(run_cache_dir, filename)
        log_q_y_cache_path = sample_cache_path.replace("data_samples_", "data_log_q_y_")
        if not os.path.exists(log_q_y_cache_path):
            continue
        cached = _load_data_sample_cache(sample_cache_path)
        if cached is None:
            continue
        candidates.append((seq_len, sample_cache_path, log_q_y_cache_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, sample_cache_path, log_q_y_cache_path = candidates[0]
    return sample_cache_path, log_q_y_cache_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="N values to ensure in existing direct/data cache",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="Match plot_bipartite_mi --max-n used for cache key derivation",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["validation", "train", "test"],
        default="validation",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
    )
    parser.add_argument(
        "--build-missing-sample-cache",
        action="store_true",
        help=(
            "If target seq data sample cache is missing, build it once from full split "
            "and then compute only missing requested N values."
        ),
    )
    args = parser.parse_args()

    requested_n_values = sorted({int(n) for n in args.n_values if int(n) >= 2})
    if not requested_n_values:
        raise RuntimeError("--n-values must include at least one integer >= 2")

    seq_len_required = _seq_len_required(int(args.max_n), requested_n_values)
    api = wandb.Api()
    runs = _resolve_group_runs(api, args.group)
    runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, args.group)

    updated_runs = 0
    skipped_no_cache = 0
    skipped_complete = 0

    for run in runs:
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])
        bos_token_id = int(cfg.get("bos_token_id", 0))

        data_key = _data_cache_key(
            split=args.data_split,
            seq_len=seq_len_required,
            num_samples=None,
            bos_token_id=bos_token_id,
            seed=None,
        )
        sample_cache_path, log_q_y_cache_path = _data_cache_paths(
            args.cache_dir,
            run.id,
            data_key,
        )

        cached = _load_data_sample_cache(sample_cache_path)

        state = None
        sample_params = None
        if cached is None and args.build_missing_sample_cache:
            print(
                "build hidden_dim="
                f"{hidden_dim}: creating missing sample cache {sample_cache_path}"
            )
            ckpt_path = _download_checkpoint_artifact(run.id, api, args.cache_dir)
            model = _build_language_model_from_config(cfg)
            rng = jax.random.PRNGKey(0)
            state_cfg = SimpleNamespace(
                batch_size=int(cfg.get("batch_size", 1)),
                seq_len=int(cfg.get("seq_len", seq_len_required)),
                learning_rate=float(cfg.get("learning_rate", 1e-3)),
            )
            state = create_train_state(model, state_cfg, rng)
            state, restored = _load_checkpoint(ckpt_path, state)
            ckpt_run_id = restored.get("wandb_run_id")
            if ckpt_run_id != run.id:
                raise RuntimeError(
                    f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
                )

            sample_params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))
            samples_full = _load_data_chunks(
                cfg=cfg,
                split=args.data_split,
                num_samples=None,
                rng=None,
            )
            samples = np.array(samples_full[:, :seq_len_required], dtype=np.int32)
            sample_logps = _score_logps(
                apply_fn=state.apply_fn,
                params=sample_params,
                tokens=samples,
                batch_size=int(args.batch_size),
                bos_token_id=bos_token_id,
                progress_desc=f"Scoring data samples d_h={hidden_dim}",
            )
            _save_data_sample_cache(sample_cache_path, samples, sample_logps)
            cached = (samples, sample_logps)

        if cached is None:
            print(
                "skip hidden_dim="
                f"{hidden_dim}: missing data sample cache {sample_cache_path}"
            )
            skipped_no_cache += 1
            continue

        samples, sample_logps = cached
        log_q_y_means_by_n = _load_log_q_y_mean_cache(log_q_y_cache_path)

        seed_paths = _find_best_seed_log_cache(
            cache_dir=args.cache_dir,
            run_id=run.id,
            split=args.data_split,
            bos_token_id=bos_token_id,
            target_seq_len=int(sample_logps.shape[1]),
        )
        if seed_paths is not None:
            _, seed_log_q_path = seed_paths
            seed_log_q_by_n = _load_log_q_y_mean_cache(seed_log_q_path)
            seeded = 0
            for n, value in seed_log_q_by_n.items():
                n_int = int(n)
                if n_int <= int(sample_logps.shape[1]) and n_int not in log_q_y_means_by_n:
                    log_q_y_means_by_n[n_int] = float(value)
                    seeded += 1
            if seeded > 0:
                print(
                    "seed hidden_dim="
                    f"{hidden_dim}: copied {seeded} N values from lower-seq cache"
                )

        out_of_range_n_values = [
            int(n) for n in requested_n_values if int(n) > int(sample_logps.shape[1])
        ]
        missing_n_values = [
            int(n)
            for n in requested_n_values
            if int(n) <= int(sample_logps.shape[1]) and int(n) not in log_q_y_means_by_n
        ]
        if not missing_n_values:
            if out_of_range_n_values:
                print(
                    "skip hidden_dim="
                    f"{hidden_dim}: requested N out of range for cached seq_len="
                    f"{sample_logps.shape[1]} -> {out_of_range_n_values}"
                )
            else:
                print(f"ok hidden_dim={hidden_dim}: all requested N already cached")
            skipped_complete += 1
            continue

        print(
            "backfill hidden_dim="
            f"{hidden_dim}: computing missing N values {missing_n_values}"
        )

        if state is None or sample_params is None:
            ckpt_path = _download_checkpoint_artifact(run.id, api, args.cache_dir)
            model = _build_language_model_from_config(cfg)
            rng = jax.random.PRNGKey(0)
            state_cfg = SimpleNamespace(
                batch_size=int(cfg.get("batch_size", 1)),
                seq_len=int(cfg.get("seq_len", seq_len_required)),
                learning_rate=float(cfg.get("learning_rate", 1e-3)),
            )
            state = create_train_state(model, state_cfg, rng)
            state, restored = _load_checkpoint(ckpt_path, state)
            ckpt_run_id = restored.get("wandb_run_id")
            if ckpt_run_id != run.id:
                raise RuntimeError(
                    f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
                )
            sample_params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))

        computed_missing = _compute_log_q_y_means(
            samples=samples,
            apply_fn=state.apply_fn,
            params=sample_params,
            n_values=missing_n_values,
            batch_size=int(args.batch_size),
            bos_token_id=bos_token_id,
            progress_desc_prefix=f"Backfill y d_h={hidden_dim}",
        )
        log_q_y_means_by_n.update(computed_missing)
        _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)
        updated_runs += 1

    print(
        "done: updated_runs="
        f"{updated_runs}, already_complete={skipped_complete}, missing_cache={skipped_no_cache}"
    )


if __name__ == "__main__":
    main()

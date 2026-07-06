"""Plot MI saturation value vs hidden_dim.

Supports either model-sampled sequences or cached dataset chunks
(see analysis/plot_bipartite_mi.py --sample-source).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import termios
import tty

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import wandb
from analysis.plot_bipartite_mi import (
    DEFAULT_MAX_N,
    DEFAULT_MIN_N,
    DEFAULT_N_VALUES,
    _compute_lstm_direct_mi_for_run_from_data,
    _compute_lstm_sampled_mi_for_run,
    _filter_runs_by_hidden_dim,
    _resolve_group_runs,
)
from analysis.plot_group_labels import distinct_group_labels

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


def _mi_saturation_value(
    series: dict[int, float], saturation_n_values: list[int]
) -> float:
    if not saturation_n_values:
        raise RuntimeError("No saturation N values provided")
    values: list[float] = []
    for n in saturation_n_values:
        if n not in series:
            raise RuntimeError(f"Missing MI value at N={n}")
        value = float(series[n])
        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite MI value at N={n}")
        values.append(value)
    return float(np.mean(np.array(values, dtype=float)))


def _plot_mi_saturation(
    rows: list[dict[str, float]],
    out_path: str,
    title: str,
    saturation_n_values: list[int],
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot")

    hidden_dims = np.array([row["hidden_dim"] for row in rows], dtype=float)
    mi_sat = np.array([row["mi_sat"] for row in rows], dtype=float)
    groups = [str(row.get("group", "default")) for row in rows]
    unique_groups = sorted(set(groups))
    group_labels = (
        distinct_group_labels(unique_groups) if len(unique_groups) > 1 else {}
    )
    show_group_name = len(unique_groups) > 1
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_group = {
        g: marker_cycle[idx % len(marker_cycle)] for idx, g in enumerate(unique_groups)
    }

    fig, ax = plt.subplots(figsize=(8, 7.5))
    for group_name in unique_groups:
        mask_group = np.array([g == group_name for g in groups], dtype=bool)
        ax.scatter(
            hidden_dims[mask_group],
            mi_sat[mask_group],
            edgecolor="black",
            marker=marker_by_group[group_name],
            label=group_labels.get(group_name, group_name) if show_group_name else None,
        )
    fit_mask = (
        (hidden_dims > 0)
        & (mi_sat > 0)
        & np.isfinite(hidden_dims)
        & np.isfinite(mi_sat)
    )
    for group_name in unique_groups:
        mask_group = np.array([g == group_name for g in groups], dtype=bool)
        fit_mask_group = fit_mask & mask_group
        if int(np.sum(fit_mask_group)) < 2:
            continue
        power, log_coef = np.polyfit(
            np.log(hidden_dims[fit_mask_group]),
            np.log(mi_sat[fit_mask_group]),
            1,
        )
        coef = float(np.exp(log_coef))
        fit_x = np.logspace(
            np.log10(float(np.min(hidden_dims[fit_mask_group]))),
            np.log10(float(np.max(hidden_dims[fit_mask_group]))),
            num=200,
        )
        fit_y = coef * np.power(fit_x, power)
        ax.plot(
            fit_x,
            fit_y,
            linestyle=":",
            linewidth=1.5,
            label=(
                rf"{group_labels.get(group_name, group_name)}: "
                rf"$\max(I(A:B))={coef:.3g}d_h^{{{power:.3f}}}$"
                if show_group_name
                else rf"$\max(I(A:B))={coef:.3g}d_h^{{{power:.3f}}}$"
            ),
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.14),
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$d_h$")
    ax.set_ylabel(r"$\mathrm{max}(I_{d_h}(A:B))$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_title(title)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, nargs="+", required=True)
    parser.add_argument(
        "--mi-estimator",
        type=str,
        choices=["direct"],
        default="direct",
        help="MI estimator to use for saturation extraction",
    )
    parser.add_argument(
        "--sample-source",
        type=str,
        choices=["model", "data"],
        default="model",
        help="Source of sequences for MI: model samples or cached dataset chunks",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["validation", "train", "test"],
        default="validation",
        help="Which cached dataset split to draw chunks from when --sample-source=data",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="RNG seed for selecting data chunks when --sample-source=data",
    )
    parser.add_argument(
        "--sanity-check-direct-data",
        action="store_true",
        help=(
            "When --mi-estimator direct and --sample-source data, verify that "
            "E[log q(B|A)] equals -sum L_n computed from the scored data logps, "
            "and that the cached log q(B) scalars are consistent with per-position losses."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--max-hidden-dim",
        type=int,
        default=2**11,
        help="Maximum hidden_dim to include",
    )
    parser.add_argument(
        "--saturation-n-values",
        type=int,
        nargs="+",
        default=[2048],
        help="N values whose MI average is treated as saturation",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="Maximum N to include in sampled MI curve",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of sampled sequences for sampled MI",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sampling/scoring sampled MI",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
        help="Directory for sampled MI cache",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Force regeneration of sampled caches instead of cache-only mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    args = parser.parse_args()

    estimator = "direct"

    if args.max_n < DEFAULT_MIN_N:
        raise RuntimeError(f"--max-n must be >= {DEFAULT_MIN_N}")
    if args.num_samples < 1:
        raise RuntimeError("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")
    if bool(getattr(args, "sanity_check_direct_data", False)):
        if estimator != "direct":
            raise RuntimeError(
                "--sanity-check-direct-data requires --mi-estimator direct"
            )
        if str(getattr(args, "sample_source", "model")) != "data":
            raise RuntimeError(
                "--sanity-check-direct-data requires --sample-source data"
            )

    saturation_n_values = sorted(set(int(n) for n in args.saturation_n_values))
    if not saturation_n_values:
        raise RuntimeError("--saturation-n-values must contain at least one N")
    if any(n < DEFAULT_MIN_N for n in saturation_n_values):
        raise RuntimeError(f"--saturation-n-values must be >= {DEFAULT_MIN_N}")
    if any(n > int(args.max_n) for n in saturation_n_values):
        raise RuntimeError("All --saturation-n-values must be <= --max-n")

    # Only compute the N values needed for saturation extraction.
    # This avoids triggering unnecessary cache misses at larger N.
    n_values = list(saturation_n_values)
    if not n_values:
        raise RuntimeError("No valid N values to evaluate")

    api = wandb.Api()
    groups = list(dict.fromkeys(args.group))
    rows: list[dict[str, float | str]] = []
    for group_name in groups:
        runs = _resolve_group_runs(api, group_name)
        runs = [
            run
            for run in runs
            if int((run.config or {}).get("hidden_dim", -1)) <= args.max_hidden_dim
        ]
        if not runs:
            raise RuntimeError(
                f"No finished runs found for group='{group_name}' "
                f"with hidden_dim <= {args.max_hidden_dim}"
            )
        runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, group_name)

        for run in tqdm(runs, desc=f"Runs[{group_name}]", unit="run"):
            hidden_dim = int((run.config or {})["hidden_dim"])
            if args.sample_source == "data":
                mi_series = _compute_lstm_direct_mi_for_run_from_data(
                    run,
                    api,
                    hidden_dim,
                    n_values,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                    data_split=args.data_split,
                    data_seed=args.data_seed,
                    sanity_check=bool(args.sanity_check_direct_data),
                )
            else:
                mi_series = _compute_lstm_sampled_mi_for_run(
                    run,
                    api,
                    hidden_dim,
                    n_values,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                )
            mi_sat = _mi_saturation_value(mi_series, saturation_n_values)
            rows.append(
                {
                    "group": group_name,
                    "hidden_dim": float(hidden_dim),
                    "mi_sat": mi_sat,
                }
            )

    rows.sort(key=lambda row: row["hidden_dim"])
    joined_ns = ",".join(str(n) for n in saturation_n_values)
    for row in rows:
        print(
            f"hidden_dim={row['hidden_dim']:.0f}, "
            f"{estimator}_mean_I(A:B)(N in {{{joined_ns}}})={row['mi_sat']:.6g}"
        )

    out_path = (
        args.output
        if args.output is not None
        else f"results/mi_saturation_{'_'.join(groups)}.png"
    )
    _plot_mi_saturation(
        rows,
        out_path,
        title=f"{','.join(groups)} ({estimator})",
        saturation_n_values=saturation_n_values,
    )


if __name__ == "__main__":
    main()

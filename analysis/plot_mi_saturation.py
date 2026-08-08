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
    CHECKPOINT_FINAL,
    DEFAULT_MAX_N,
    DEFAULT_MIN_N,
    DEFAULT_N_VALUES,
    _compute_lstm_direct_mi_for_run_from_data,
    _compute_lstm_sampled_mi_for_run,
    _compute_with_cache_fill,
    _filter_runs_by_hidden_dim,
    _drop_tail_milestone,
    _format_token_count,
    _list_token_checkpoints,
    _normalize_split,
    _read_mi_from_wandb,
    _resolve_group_runs,
    token_checkpoint,
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


def _tail_averaged_mi_value(
    series: dict[int, float],
    *,
    seq_len: int,
) -> tuple[int, float]:
    if not series:
        raise RuntimeError("No MI values available")
    items = sorted(series.items())
    if int(seq_len) > 2048:
        tail_items = items[-min(3, len(items)) :]
        tail_ns = [int(n) for n, _ in tail_items]
        tail_values = [float(v) for _, v in tail_items]
        if not all(np.isfinite(v) for v in tail_values):
            raise RuntimeError(
                f"Non-finite MI values in tail for seq_len={seq_len}: {tail_ns}"
            )
        return tail_ns[-1], float(np.mean(tail_values))

    largest_n = int(items[-1][0])
    value = float(items[-1][1])
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite MI value at N={largest_n}")
    return largest_n, value


def _plot_mi_saturation(
    rows: list[dict[str, float]],
    out_path: str,
    title: str,
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot")

    unique_groups = sorted({str(row.get("group", "default")) for row in rows})
    group_labels = (
        distinct_group_labels(unique_groups) if len(unique_groups) > 1 else {}
    )
    show_group_name = len(unique_groups) > 1
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_group = {
        g: marker_cycle[idx % len(marker_cycle)] for idx, g in enumerate(unique_groups)
    }

    # One series per (group, token checkpoint). With token checkpoints the
    # colour scale encodes training tokens, exactly as in plot_bipartite_mi, so
    # a whole family of saturation curves fits on one axes.
    token_values = sorted({r["tokens"] for r in rows if r.get("tokens") is not None})
    cmap = plt.cm.viridis
    if token_values:
        norm = plt.Normalize(
            vmin=(token_values[0] * 0.99 if len(token_values) == 1 else token_values[0]),
            vmax=(token_values[-1] * 1.01 if len(token_values) == 1 else token_values[-1]),
        )

    def _series_color(tokens):
        return cmap(norm(tokens)) if token_values and tokens is not None else None

    series: dict[tuple[str, float | None], list[dict]] = {}
    for row in rows:
        key = (str(row.get("group", "default")), row.get("tokens"))
        series.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    for (group_name, tokens), series_rows in sorted(
        series.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)
    ):
        series_rows = sorted(series_rows, key=lambda r: r["hidden_dim"])
        hd = np.array([r["hidden_dim"] for r in series_rows], dtype=float)
        sat = np.array([r["mi_sat"] for r in series_rows], dtype=float)
        color = _series_color(tokens)
        label_parts = []
        if show_group_name:
            label_parts.append(group_labels.get(group_name, group_name))
        if tokens is not None:
            label_parts.append(f"{_format_token_count(int(tokens))} tokens")
        ax.scatter(
            hd,
            sat,
            edgecolor="black",
            marker=marker_by_group[group_name],
            color=color,
            label=", ".join(label_parts) if label_parts else None,
        )

        fit_mask = (hd > 0) & (sat > 0) & np.isfinite(hd) & np.isfinite(sat)
        if int(np.sum(fit_mask)) < 2:
            continue
        power, log_coef = np.polyfit(np.log(hd[fit_mask]), np.log(sat[fit_mask]), 1)
        coef = float(np.exp(log_coef))
        fit_x = np.logspace(
            np.log10(float(np.min(hd[fit_mask]))),
            np.log10(float(np.max(hd[fit_mask]))),
            num=200,
        )
        prefix = ", ".join(label_parts)
        ax.plot(
            fit_x,
            coef * np.power(fit_x, power),
            linestyle=":",
            linewidth=1.5,
            color=color,
            label=(
                (rf"{prefix}: " if prefix else "")
                + rf"$\max(I(A:B))={coef:.3g}d_h^{{{power:.3f}}}$"
            ),
        )
        print(
            f"fit group={group_name} "
            f"tokens={int(tokens) if tokens is not None else 'final'}: "
            f"max(I(A:B))={coef:.4g}*d_h^{power:.4f} "
            f"({int(np.sum(fit_mask))} points)"
        )

    if token_values:
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02
        )
        colorbar.set_label("training tokens")
        colorbar.set_ticks(token_values)
        colorbar.set_ticklabels([_format_token_count(int(t)) for t in token_values])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$d_h$")
    ax.set_ylabel(r"$\mathrm{max}(I_{d_h}(A:B))$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # bbox_inches="tight" rather than tight_layout: it crops to the rendered
    # artists, so anything drawn outside the axes rectangle (the colorbar here)
    # is included rather than clipped.
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
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
        choices=["validation", "train", "test", "validation+test", "test+validation", "train_tail"],
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
        "--from-wandb",
        action="store_true",
        help=(
            "Read the N -> I(A:B) series from each run's W&B summary instead of "
            "scoring locally; saturation is then the tail average of that series."
        ),
    )
    parser.add_argument(
        "--all-token-checkpoints",
        action="store_true",
        help=(
            "Plot one saturation curve per token milestone "
            "(checkpoint-<run_id>-tokens-<n>), discovered from the runs' logged "
            "artifacts, coloured by training tokens. For runs trained with "
            "checkpoint_every_n_tokens > 0."
        ),
    )
    parser.add_argument(
        "--checkpoint-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Explicit token milestones, instead of --all-token-checkpoints.",
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

    if args.all_token_checkpoints and args.checkpoint_tokens:
        raise RuntimeError(
            "--all-token-checkpoints and --checkpoint-tokens are mutually exclusive"
        )

    api = wandb.Api()
    groups = list(dict.fromkeys(args.group))
    runs_by_group: dict[str, list] = {}
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
        runs_by_group[group_name] = _filter_runs_by_hidden_dim(
            runs, args.hidden_dim, group_name
        )

    milestones_by_run: dict[str, set[int]] = {}
    if args.all_token_checkpoints or args.checkpoint_tokens:
        for group_runs in runs_by_group.values():
            for run in group_runs:
                milestones_by_run[run.id] = set(_list_token_checkpoints(run))
    if args.checkpoint_tokens:
        milestones = sorted(dict.fromkeys(int(n) for n in args.checkpoint_tokens))
    elif args.all_token_checkpoints:
        milestones = _drop_tail_milestone(
            sorted(set().union(*milestones_by_run.values()))
        )
        if not milestones:
            raise RuntimeError(
                "--all-token-checkpoints found no checkpoint-<run_id>-tokens-<n> "
                "artifacts. Only runs trained with checkpoint_every_n_tokens > 0 "
                "have them."
            )
        print(
            f"Found {len(milestones)} token milestones: "
            + ", ".join(_format_token_count(n) for n in milestones)
        )
    else:
        milestones = []
    checkpoints = (
        [(token_checkpoint(n), float(n)) for n in milestones]
        if milestones
        else [(CHECKPOINT_FINAL, None)]
    )

    rows: list[dict[str, float | str]] = []
    for group_name in groups:
        for run in tqdm(runs_by_group[group_name], desc=f"Runs[{group_name}]", unit="run"):
            hidden_dim = int((run.config or {})["hidden_dim"])
            run_seq_len = int((run.config or {}).get("seq_len", int(args.max_n)))
            n_values = [n for n in DEFAULT_N_VALUES if n <= run_seq_len]
            if not n_values:
                print(
                    "Skipping hidden_dim="
                    f"{hidden_dim}: no valid N values for seq_len={run_seq_len}"
                )
                continue
            for checkpoint, tokens in checkpoints:
                if (
                    tokens is not None
                    and run.id in milestones_by_run
                    and int(tokens) not in milestones_by_run[run.id]
                ):
                    print(
                        f"Skipping hidden_dim={hidden_dim}: run {run.id} has no "
                        f"checkpoint at {_format_token_count(int(tokens))} tokens"
                    )
                    continue
                desc = f"hidden_dim={hidden_dim} checkpoint={checkpoint}"
                # Compute on a cache miss rather than silently dropping the
                # point. Reading cache-only meant a run that had never been
                # scored just vanished from the plot with a one-line notice,
                # and the remaining points still produced a confident fit.
                if args.from_wandb:
                    mi_series = _read_mi_from_wandb(
                        run,
                        estimator=estimator,
                        data_split=_normalize_split(args.data_split),
                        checkpoint=checkpoint,
                    )
                elif args.sample_source == "data":
                    mi_series = _compute_with_cache_fill(
                        lambda force: _compute_lstm_direct_mi_for_run_from_data(
                            run,
                            api,
                            hidden_dim,
                            n_values,
                            batch_size=args.batch_size,
                            cache_dir=args.cache_dir,
                            force_resample=force,
                            data_split=args.data_split,
                            sanity_check=bool(args.sanity_check_direct_data),
                            checkpoint=checkpoint,
                        ),
                        force_resample=args.force_resample,
                        fill_missing=True,
                        desc=f"direct/data {desc}",
                    )
                else:
                    mi_series = _compute_with_cache_fill(
                        lambda force: _compute_lstm_sampled_mi_for_run(
                            run,
                            api,
                            hidden_dim,
                            n_values,
                            num_samples=args.num_samples,
                            batch_size=args.batch_size,
                            cache_dir=args.cache_dir,
                            force_resample=force,
                            checkpoint=checkpoint,
                        ),
                        force_resample=args.force_resample,
                        fill_missing=True,
                        desc=f"direct/model {desc}",
                    )
                if not mi_series:
                    print(
                        "Skipping hidden_dim="
                        f"{hidden_dim} checkpoint={checkpoint}: "
                        "no cached MI values available"
                    )
                    continue
                saturation_n, mi_sat = _tail_averaged_mi_value(
                    mi_series,
                    seq_len=run_seq_len,
                )
                rows.append(
                    {
                        "group": group_name,
                        "hidden_dim": float(hidden_dim),
                        "mi_sat": mi_sat,
                        "saturation_n": float(saturation_n),
                        "tokens": tokens,
                    }
                )

    rows.sort(key=lambda row: (row.get("tokens") or 0, row["hidden_dim"]))
    for row in rows:
        tokens = row.get("tokens")
        label = f"{_format_token_count(int(tokens))} tokens, " if tokens else ""
        print(
            f"{label}hidden_dim={row['hidden_dim']:.0f}, "
            f"{estimator}_I(A:B)(N={int(row['saturation_n'])})={row['mi_sat']:.6g}"
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
    )


if __name__ == "__main__":
    main()

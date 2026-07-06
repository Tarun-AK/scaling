"""Plot a scaling collapse of n-gram losses.

The plot uses n-gram losses from `plot_ngrams.py` and a tail-average estimate of
L_infinity, then visualizes

  y = (L_n - L_infinity) / n ** (1 - x_1)
  x = hidden_dim / n ** ((2 - x_1) / alpha)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import termios
import tty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb
from analysis.plot_ngrams import extract_metrics

plt.style.use("~/plotStyle.mplstyle")

TAIL_POINTS = 20


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


def fetch_runs(project: str, group: str | None = None) -> list[wandb.apis.public.Run]:
    api = wandb.Api(timeout=60)
    filters = {"group": group, "state": "finished"} if group else {"state": "finished"}
    runs = api.runs(project, filters=filters)
    return list(runs)


def _tail_average(values: np.ndarray, tail_points: int = TAIL_POINTS) -> float:
    if values.size == 0:
        raise RuntimeError("Cannot compute tail average from empty values")
    tail_n = min(int(tail_points), int(values.size))
    if tail_n < 1:
        raise RuntimeError("tail_points must be >= 1")
    return float(np.mean(values[-tail_n:]))


def _prepare_dataframe(runs: list[wandb.apis.public.Run]) -> pd.DataFrame:
    df = extract_metrics(runs, split="combined")
    if df.empty:
        return df

    df = df.groupby(["group", "hidden_dim", "n"], as_index=False)["loss"].mean()
    df = df.sort_values(["group", "hidden_dim", "n"])

    rows: list[dict[str, float | int | str]] = []
    for (group_name, hidden_dim), group_df in df.groupby(["group", "hidden_dim"]):
        group_df = group_df.sort_values("n")
        losses = group_df["loss"].to_numpy(dtype=float)
        l_inf = _tail_average(losses)
        for _, row in group_df.iterrows():
            n = float(row["n"])
            loss = float(row["loss"])
            if not np.isfinite(n) or n <= 0:
                continue
            rows.append(
                {
                    "group": str(group_name),
                    "hidden_dim": int(hidden_dim),
                    "n": int(row["n"]),
                    "loss": loss,
                    "l_inf": l_inf,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["group", "hidden_dim", "n", "loss", "l_inf"])
    return pd.DataFrame(rows).sort_values(["group", "hidden_dim", "n"])


def _filter_n_range(
    df: pd.DataFrame,
    min_n: int | None,
    max_n: int | None,
) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if min_n is not None:
        mask &= df["n"] >= int(min_n)
    if max_n is not None:
        mask &= df["n"] <= int(max_n)
    return df[mask].copy()


def plot_scaling_collapse(
    df: pd.DataFrame,
    out_path: str,
    x_1: float,
    alpha: float,
    group: str,
) -> None:
    if df.empty:
        raise RuntimeError("No completed runs with required metrics found.")

    fig, ax = plt.subplots(figsize=(8, 7.5))
    hidden_dims = sorted(df["hidden_dim"].unique())
    cmap = plt.get_cmap("viridis")
    if len(hidden_dims) > 1:
        norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
    else:
        norm = plt.Normalize(vmin=hidden_dims[0], vmax=hidden_dims[0] + 1)

    x_power = (2.0 - float(x_1)) / float(alpha)
    y_power = 1.0 - float(x_1)

    for hidden_dim, hidden_df in df.groupby("hidden_dim"):
        hidden_df = hidden_df.sort_values("n")
        n_values = hidden_df["n"].to_numpy(dtype=float)
        losses = hidden_df["loss"].to_numpy(dtype=float)
        l_inf = float(hidden_df["l_inf"].iloc[0])

        x_vals = hidden_dim / np.power(n_values, x_power)
        y_vals = (losses - l_inf) / np.power(n_values, y_power)

        valid = (
            np.isfinite(x_vals)
            & np.isfinite(y_vals)
            & (x_vals > 0)
            & (y_vals > 0)
        )
        if not np.any(valid):
            continue

        ax.plot(
            x_vals[valid],
            y_vals[valid],
            marker="o",
            linewidth=1.5,
            alpha=0.85,
            color=cmap(norm(hidden_dim)),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$d_h / n^{{(2 - x_1)/\alpha}}$")
    ax.set_ylabel(rf"$(L_n - L_\infty)/n^{{1 - x_1}}$")
    ax.set_title(
        rf"Scaling collapse ({group}), $x_1={x_1:.3g}$, $\alpha={alpha:.3g}$"
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$d_h$")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_1", "--x-1", dest="x_1", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--min-n", type=int, default=None)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    runs = fetch_runs("scaling", group=args.group)
    df = _filter_n_range(_prepare_dataframe(runs), args.min_n, args.max_n)
    out_path = (
        args.output
        if args.output is not None
        else f"results/scaling_collapse_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', args.group)}.png"
    )
    plot_scaling_collapse(df, out_path, x_1=args.x_1, alpha=args.alpha, group=args.group)


if __name__ == "__main__":
    main()

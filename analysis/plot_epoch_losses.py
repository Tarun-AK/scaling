"""Plot training vs validation loss across epochs.

Usage:
  python analysis/plot_epoch_losses.py --group seq_len_128
  python analysis/plot_epoch_losses.py --group seq_len_128 --hidden-dims 128 256
  python analysis/plot_epoch_losses.py --hidden-dims 128,256,512
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import termios
import tty
import os
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import wandb

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


def _parse_hidden_dims(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    dims: list[int] = []
    for val in values:
        for part in val.split(","):
            part = part.strip()
            if part:
                dims.append(int(part))
    return sorted(set(dims)) if dims else None


def fetch_runs(project: str, group: str | None = None) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    filters = {"group": group} if group else {}
    runs = api.runs(project, filters=filters)
    print(f"Found {len(runs)} runs total")
    return [r for r in runs if r.state in ("finished", "running")]


def extract_epoch_losses(
    runs: List[wandb.apis.public.Run],
    hidden_dims: list[int] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in runs:
        cfg = r.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        hidden_dim = int(hidden_dim)
        if hidden_dims is not None and hidden_dim not in hidden_dims:
            continue

        train_col = "train/loss"
        summary_keys = list((r.summary or {}).keys())
        val_cols = [k for k in summary_keys if k.startswith("val/ngram_")]
        if not val_cols:
            history_cols = list(r.history(samples=1).columns)
            val_cols = [k for k in history_cols if k.startswith("val/ngram_")]
        if not val_cols:
            continue

        val_df = r.history(keys=val_cols + ["_step"], samples=10000)
        train_df = r.history(keys=[train_col, "_step"], samples=10000)
        if val_df.empty or train_df.empty:
            continue
        if "_step" not in val_df.columns or "_step" not in train_df.columns:
            continue

        val_df = val_df.dropna(how="all", subset=val_cols).sort_values("_step")
        if val_df.empty:
            continue
        prev_step = -1
        train_df = train_df.sort_values("_step")
        for epoch_idx, (_, val_row) in enumerate(val_df.iterrows()):
            step_val = val_row.get("_step")
            if pd.isna(step_val):
                continue
            step = int(step_val)
            val_loss = float(pd.Series(val_row[val_cols]).mean())

            train_mask = (train_df["_step"] > prev_step) & (train_df["_step"] <= step)
            train_vals = train_df.loc[train_mask, train_col].dropna()
            train_loss = float(train_vals.mean()) if len(train_vals) > 0 else None

            rows.append(
                {
                    "run": r.id,
                    "hidden_dim": hidden_dim,
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            prev_step = step

    if not rows:
        return pd.DataFrame(
            columns=["run", "hidden_dim", "epoch", "train_loss", "val_loss"]
        )

    df = pd.DataFrame(rows)
    grouped = df.groupby(["hidden_dim", "epoch"], as_index=False)[
        ["train_loss", "val_loss"]
    ].mean()
    return grouped.sort_values(["hidden_dim", "epoch"])


def plot_epoch_losses(
    df: pd.DataFrame,
    out_path: str,
    group: str | None,
) -> None:
    if df.empty:
        raise RuntimeError("No data found.")

    plt.figure(figsize=(10, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    hidden_dim_handles: list[Line2D] = []

    for i, (hidden_dim, group_df) in enumerate(df.groupby("hidden_dim")):
        color = colors[i % len(colors)]
        group_df = group_df.sort_values("epoch")

        train_df = group_df.dropna(subset=["train_loss"])
        plt.plot(
            train_df["epoch"],
            train_df["train_loss"],
            color=color,
            marker="o",
            label=f"train (d_h={hidden_dim})",
        )
        plt.plot(
            group_df["epoch"],
            group_df["val_loss"],
            color=color,
            linestyle="--",
            marker="s",
            label=None,
        )
        hidden_dim_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="-",
                label=f"d_h={hidden_dim}",
            )
        )

    if group:
        plt.title(group)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="train"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="val"),
    ]
    style_legend = plt.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    plt.gca().add_artist(style_legend)
    plt.legend(
        handles=hidden_dim_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="W&B group to filter runs",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="*",
        default=None,
        help="Hidden dims to include (space or comma separated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/epoch_train_val_loss.png",
    )
    args = parser.parse_args()

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    runs = fetch_runs(WANDB_PROJECT, group=args.group)
    df = extract_epoch_losses(runs, hidden_dims=hidden_dims)
    plot_epoch_losses(df, args.output, args.group)


if __name__ == "__main__":
    main()

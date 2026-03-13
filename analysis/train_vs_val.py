"""Plot final train vs validation loss as a function of hidden dimension.

Usage:
    python analysis/train_vs_val.py

Produces results/overfitting.png.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb

WANDB_PROJECT = "tarunadvaith-/scaling"
TRAIN_AVG_LAST_N = 100
plt.style.use("~/plotStyle.mplstyle")


def fetch_runs(project: str) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    runs = api.runs(project)
    runs = api.runs(project, filters={"group": None})
    return [r for r in runs if r.state in ("finished", "running")]


def extract_final_losses(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    rows = []
    for r in runs:
        hidden_dim = r.config.get("hidden_dim")
        if hidden_dim is None:
            continue

        history = r.history()
        if history.empty:
            continue

        # Final train loss: mean of last TRAIN_AVG_LAST_N logged values
        train_col = "train/loss"
        if train_col not in history.columns:
            continue
        train_series = history[train_col].dropna()
        if len(train_series) == 0:
            continue
        final_train = train_series.iloc[-TRAIN_AVG_LAST_N:].mean()

        # Final val+test loss: mean of all val/ngram_* and test/ngram_* at last logged row
        val_cols = [c for c in history.columns if c.startswith("val/ngram_")]
        test_cols = [c for c in history.columns if c.startswith("test/ngram_")]
        combined_cols = val_cols + test_cols
        if not combined_cols:
            continue
        last_row = history[combined_cols].dropna(how="all").iloc[-1]
        final_eval = last_row.mean()

        rows.append(
            {
                "hidden_dim": int(hidden_dim),
                "train_loss": final_train,
                "val_loss": final_eval,
            }
        )

    return pd.DataFrame(rows).sort_values("hidden_dim")


def plot_overfitting(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        raise RuntimeError("No data found.")

    plt.figure(figsize=(7, 5))
    plt.plot(
        df["hidden_dim"],
        df["train_loss"],
        marker="o",
        label="train loss (final)",
        markeredgecolor="black",
    )
    plt.plot(
        df["hidden_dim"],
        df["val_loss"],
        marker="o",
        label="val loss (final)",
        markeredgecolor="black",
    )
    plt.xscale("log", base=2)
    plt.xlabel(r"$d_h$")
    plt.ylabel("L")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


def main() -> None:
    runs = fetch_runs(WANDB_PROJECT)
    df = extract_final_losses(runs)
    plot_overfitting(df, "results/overfitting.png")


if __name__ == "__main__":
    main()

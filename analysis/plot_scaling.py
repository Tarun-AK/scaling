"""Plot hidden-dimension scaling curves from Weights & Biases.

This script queries completed runs from the W&B project `lstm_scaling`, extracts
final test n-gram losses along with hidden_dim, and produces scaling plots saved
to results/scaling_curves.png.

Usage:
  python analysis/plot_scaling.py

You may need to login first:
  wandb login
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

import wandb

PLOT_POSITIONS = [1, 2, 5, 10, 15, 20, 25, 31]


def fetch_runs(project: str) -> List[wandb.apis.public.Run]:
    """Fetch completed runs from a W&B project."""
    api = wandb.Api()
    runs = api.runs(project)
    return [r for r in runs if r.state == "finished"]


def extract_metrics(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    """Extract hidden_dim and final test n-gram losses into a DataFrame.

    Only extracts losses at positions defined in PLOT_POSITIONS.
    """
    rows: List[Dict[str, Any]] = []
    for r in runs:
        cfg = r.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        summary = r.summary or {}
        row: Dict[str, Any] = {"hidden_dim": int(hidden_dim)}
        for n in PLOT_POSITIONS:
            val = summary.get(f"test/ngram_{n}")
            if val is None:
                break
            row[f"ngram_{n}"] = val
        else:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("hidden_dim")
    return df


def plot_scaling(df: pd.DataFrame, out_path: str) -> None:
    """Plot scaling curves for each n-gram position and save figure."""
    if df.empty:
        raise RuntimeError("No completed runs with required metrics found.")

    plt.figure(figsize=(8, 5))
    for n in PLOT_POSITIONS:
        col = f"ngram_{n}"
        if col in df.columns:
            plt.plot(df["hidden_dim"], df[col], marker="o", label=f"n={n}")

    plt.xscale("log", base=2)
    plt.xlabel("Hidden dimension")
    plt.ylabel("Test n-gram cross-entropy")
    plt.title("LSTM hidden_dim scaling on WikiText")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


def main() -> None:
    runs = fetch_runs("scaling")
    df = extract_metrics(runs)
    plot_scaling(df, "results/scaling_curves.png")


if __name__ == "__main__":
    main()

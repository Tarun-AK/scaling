"""Plot n-gram loss as a function of position n for each hidden_dim run.

Usage:
  python analysis/plot_ngrams.py

Produces results/ngram_curves.png.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import wandb


def power_law(n, L_inf, c, power):
    return L_inf + c * np.power(n, power)


plt.style.use("~/plotStyle.mplstyle")


def fetch_runs(project: str) -> List[wandb.apis.public.Run]:
    """Fetch completed runs from a W&B project."""
    api = wandb.Api()
    runs = api.runs(project)
    return [r for r in runs if r.state == "finished"]


def extract_metrics(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    """Extract per-position test n-gram losses for each run."""
    rows: List[Dict[str, Any]] = []
    for r in runs:
        cfg = r.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        summary = r.summary or {}
        # Collect all test/ngram_* keys
        ngram_vals = {}
        for k, v in summary.items():
            if k.startswith("combined/ngram_"):
                n = int(k.split("_")[-1])
                ngram_vals[n] = v
        if not ngram_vals:
            continue
        for n, loss in ngram_vals.items():
            rows.append({"hidden_dim": int(hidden_dim), "n": n, "loss": loss})
    return pd.DataFrame(rows).sort_values(["hidden_dim", "n"])


def plot_ngrams(
    df: pd.DataFrame,
    out_path: str,
    xlim: tuple[int, int] | None = None,
    fit_nmax: int = 50,
) -> None:
    """Plot L_n - L_inf vs n, one line per hidden_dim, with L_inf from power-law fit on n <= fit_nmax."""
    if df.empty:
        raise RuntimeError("No completed runs with required metrics found.")

    plt.figure(figsize=(9, 9))

    legend_entries: List[tuple[int, float, float]] = []

    for hidden_dim, group in df.groupby("hidden_dim"):
        group = group.sort_values("n")
        ns = group["n"].to_numpy(dtype=float)
        losses = group["loss"].to_numpy(dtype=float)

        # Fit only on clean low-n region
        fit_mask = ns <= fit_nmax
        ns_fit, losses_fit = ns[fit_mask], losses[fit_mask]

        try:
            p0 = [losses_fit[-1], losses_fit[0] - losses_fit[-1], -0.5]
            popt, _ = curve_fit(
                power_law,
                ns_fit,
                losses_fit,
                p0=p0,
                maxfev=10_000,
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
            )
            L_inf, c, power = popt
            print(f"hd={hidden_dim}: L_inf={L_inf:.4f}, c={c:.4f}, power={power:.4f}")
            legend_entries.append((int(hidden_dim), power, L_inf))
            fit_success = True
        except RuntimeError as e:
            print(
                f"Fit failed for hidden_dim={hidden_dim}: {e}. Falling back to empirical min."
            )
            L_inf = np.min(losses_fit)
            fit_success = False

        cmi = losses - L_inf

        (line,) = plt.plot(
            ns[cmi > 1e-4],
            cmi[cmi > 1e-4],
            marker="o",
            markeredgecolor="black",
            label=f"hd={hidden_dim}",
        )
        color = line.get_color()

        if fit_success:
            ns_fit = np.linspace(1, ns[-1], 200)
            fit_curve = power_law(ns_fit, L_inf, c, power) - L_inf
            plt.plot(
                ns_fit,
                fit_curve,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

    plt.xlabel("n")
    plt.ylabel(r"$L_n - L_{\infty}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("N-gram loss vs position")
    if xlim is not None:
        plt.xlim(xlim)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    legend_labels = [
        rf"$d_{{h}}={hd} \,\, \gamma={power:.3f}$"
        for hd, power, L_inf in sorted(legend_entries)
    ]
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles,
        legend_labels,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlim",
        type=int,
        nargs=2,
        default=None,
        help="X-axis range as two integers (e.g., 1 20)",
    )
    args = parser.parse_args()
    xlim = tuple(args.xlim) if args.xlim else None

    runs = fetch_runs("scaling")
    df = extract_metrics(runs)
    plot_ngrams(df, "results/ngram_curves.png", xlim=xlim)


if __name__ == "__main__":
    main()

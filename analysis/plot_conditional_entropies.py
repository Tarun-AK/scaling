"""Plot conditional entropies as a function of position n for each hidden_dim run.

Usage:
  python analysis/plot_conditional_entropies.py

Produces results/conditional_entropy_curves.png.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import wandb


def power_law(n, L_inf, c, power):
    return L_inf + c * np.power(n, power)


plt.style.use("~/plotStyle.mplstyle")


def fetch_runs(project: str) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    runs = api.runs(project)
    runs = api.runs(project, filters={"group": None})
    return [r for r in runs if r.state == "finished"]


def extract_metrics(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        cfg = r.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        summary = r.summary
        entropy_keys = [
            k for k in summary.keys() if k.startswith("conditional_entropy/entropy_")
        ]
        if not entropy_keys:
            continue
        history = r.history()
        if history is None or history.empty:
            continue
        for _, row in history.iterrows():
            step = int(row["_step"])
            epoch_val = row.get("epoch")
            epoch = int(epoch_val) if pd.notna(epoch_val) else step
            for k, v in row.items():
                if k.startswith("conditional_entropy/entropy_") and pd.notna(v):
                    n = int(k.split("_")[-1])
                    rows.append(
                        {
                            "hidden_dim": int(hidden_dim),
                            "epoch": epoch,
                            "step": step,
                            "n": n,
                            "loss": float(v),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["hidden_dim", "epoch", "step", "n", "loss"])
    return pd.DataFrame(rows).sort_values(["hidden_dim", "epoch", "n"])


def plot_conditional_entropies(
    df: pd.DataFrame,
    out_path: str,
    xlim: tuple[int, int] | None = None,
    fit_nmax: int = 30,
    group_by_epoch: bool = False,
) -> None:
    if df.empty:
        raise RuntimeError("No completed runs with required metrics found.")

    plt.figure(figsize=(9, 7))

    if group_by_epoch:
        unique_steps = sorted(df["step"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_steps)))
        step_to_color = {step: colors[i] for i, step in enumerate(unique_steps)}

        for step, group in df.groupby("step"):
            group = group.sort_values("n")
            ns = group["n"].to_numpy(dtype=float)
            losses = group["loss"].to_numpy(dtype=float)

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
                print(f"step={step}: L_inf={L_inf:.4f}, c={c:.4f}, power={power:.4f}")
            except RuntimeError as e:
                print(
                    f"Fit failed for step={step}: {e}. Falling back to empirical min."
                )
                L_inf = np.min(losses_fit)

            adjusted = losses

            plt.plot(
                ns,
                adjusted,
                color=step_to_color[step],
                marker="o",
                markeredgecolor="black",
                label=f"t = {step}",
                alpha=0.8,
            )
        plt.legend(loc="best", fontsize=8)
    else:
        for hidden_dim, group in df.groupby("hidden_dim"):
            group = group.sort_values("n")
            ns = group["n"].to_numpy(dtype=float)
            losses = group["loss"].to_numpy(dtype=float)

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
                print(
                    f"hd={hidden_dim}: L_inf={L_inf:.4f}, c={c:.4f}, power={power:.4f}"
                )
            except RuntimeError as e:
                print(
                    f"Fit failed for hidden_dim={hidden_dim}: {e}. Falling back to empirical min."
                )
                L_inf = np.min(losses_fit)

            cmi = losses - L_inf

            plt.plot(
                ns,
                cmi,
                marker="o",
                markeredgecolor="black",
                label=f"hd={hidden_dim}",
            )
        plt.legend()

    plt.xlabel("n")
    plt.ylabel(r"$H_n$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Conditional entropy vs position")
    if xlim is not None:
        plt.xlim(xlim)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
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
    parser.add_argument(
        "--group-by-epoch",
        action="store_true",
        help="Plot separate lines for each epoch",
    )
    args = parser.parse_args()

    runs = fetch_runs("tarunadvaith-/scaling")
    df = extract_metrics(runs)
    plot_conditional_entropies(
        df,
        "results/conditional_entropy_curves.png",
        xlim=tuple(args.xlim) if args.xlim else None,
        group_by_epoch=args.group_by_epoch,
    )


if __name__ == "__main__":
    main()

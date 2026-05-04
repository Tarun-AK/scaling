"""Plot conditional entropies as a function of position n for each run.

Usage:
  python analysis/plot_conditional_entropies.py --group <group>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import termios
import tty
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import wandb


def power_law(n, l_inf, c, power):
    return l_inf + c * np.power(n, power)


plt.style.use("~/plotStyle.mplstyle")
Y_LIM_MIN = float(np.sqrt(1e-3 * 1e-2))


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


def fetch_runs(project: str, group: str | None = None) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    filters = {"group": group} if group else {}
    runs = api.runs(project, filters=filters)
    return [r for r in runs if r.state == "finished"]


def extract_metrics(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        cfg = run.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue

        summary = run.summary or {}
        entropy_cols = [
            key
            for key in summary.keys()
            if str(key).startswith("conditional_entropy/entropy_")
        ]

        if entropy_cols:
            epoch_val = summary.get("epoch", 0)
            epoch = int(epoch_val) if pd.notna(epoch_val) else 0
            for key in entropy_cols:
                n = int(str(key).split("_")[-1])
                value = summary.get(key)
                if not pd.notna(value):
                    continue
                rows.append(
                    {
                        "run": run.id,
                        "hidden_dim": int(hidden_dim),
                        "epoch": epoch,
                        "n": n,
                        "entropy": float(value),
                    }
                )
            continue

        history = run.history()
        if history is None or history.empty:
            continue

        for _, row in history.iterrows():
            epoch_val = row.get("epoch")
            epoch = int(epoch_val) if pd.notna(epoch_val) else int(row.get("_step", 0))
            for key, value in row.items():
                if str(key).startswith("conditional_entropy/entropy_") and pd.notna(
                    value
                ):
                    n = int(str(key).split("_")[-1])
                    rows.append(
                        {
                            "run": run.id,
                            "hidden_dim": int(hidden_dim),
                            "epoch": epoch,
                            "n": n,
                            "entropy": float(value),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["run", "hidden_dim", "epoch", "n", "entropy"])
    return pd.DataFrame(rows).sort_values(["run", "epoch", "n"])


def plot_conditional_entropies(
    df: pd.DataFrame,
    out_path: str,
    group: str | None,
    raw: bool,
    fit_nmax: int = 20,
    n_min: int | None = None,
    n_max: int | None = None,
) -> None:
    if df.empty:
        raise RuntimeError("No data found.")

    fig, ax = plt.subplots(figsize=(12, 12))

    hidden_dims = sorted(df["hidden_dim"].unique())
    norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
    cmap = plt.cm.viridis

    for hidden_dim, hd_group in df.groupby("hidden_dim"):
        group_df = (
            hd_group.groupby("n", as_index=False)["entropy"].mean().sort_values("n")
        )
        if n_min is not None:
            group_df = group_df[group_df["n"] >= n_min]
        if n_max is not None:
            group_df = group_df[group_df["n"] <= n_max]
        if group_df.empty:
            continue
        ns = group_df["n"].to_numpy(dtype=float)
        entropies = group_df["entropy"].to_numpy(dtype=float)
        fit_label = rf"$d_h={hidden_dim}$"

        if raw:
            l_inf = 0.0
            c = 0.0
            power = 0.0
            fit_success = False
            values = entropies
        else:
            fit_mask = ns <= fit_nmax
            ns_fit = ns[fit_mask]
            entropies_fit = entropies[fit_mask]

            fit_success = False
            if len(ns_fit) >= 2:
                try:
                    p0 = [entropies_fit[-1], entropies_fit[0] - entropies_fit[-1], -0.5]
                    popt, _ = curve_fit(
                        power_law,
                        ns_fit,
                        entropies_fit,
                        p0=p0,
                        maxfev=10_000,
                        bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
                    )
                    l_inf, c, power = popt
                    fit_label = (
                        rf"$d_h={hidden_dim}$: "
                        rf"$H_n={c:.3g}\cdot n^{{{power:.3f}}}+{l_inf:.3g}$"
                    )
                    fit_success = True
                except RuntimeError:
                    fit_success = False

            if not fit_success:
                if len(entropies_fit) > 0:
                    l_inf = float(np.min(entropies_fit))
                else:
                    l_inf = float(np.min(entropies))
                fit_label = rf"$d_h={hidden_dim}$: fit failed"

            values = entropies

        (line,) = ax.plot(
            ns,
            values,
            marker="o",
            markeredgecolor="black",
            linestyle="-",
            label=fit_label,
            color=cmap(norm(hidden_dim)),
            alpha=0.8,
        )

        if (not raw) and fit_success:
            ns_fit_dense = np.linspace(1, ns[-1], 200)
            fit_curve = power_law(ns_fit_dense, l_inf, c, power)
            ax.plot(
                ns_fit_dense,
                fit_curve,
                color=line.get_color(),
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
            )

    ax.set_xlabel("n")
    ax.set_ylabel(r"$H_n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if group:
        ax.set_title(group)
    else:
        ax.set_title("Conditional entropy vs position")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if len(ax.lines) == 0:
        raise RuntimeError("No data left after n-range filtering.")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=1,
        borderaxespad=0.0,
        labelspacing=0.25,
        handletextpad=0.4,
        frameon=False,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
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
    parser.add_argument("--output", type=str, default="results/conditional_entropy.png")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw conditional entropy without fitting",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=None,
        help="Minimum n to include",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Maximum n to include",
    )
    args = parser.parse_args()

    runs = fetch_runs("tarunadvaith-/scaling", group=args.group)
    df = extract_metrics(runs)
    if df.empty:
        raise RuntimeError("No data found.")

    max_epochs = df.groupby("run")["epoch"].transform("max")
    df = df[df["epoch"] == max_epochs]

    plot_conditional_entropies(
        df,
        args.output,
        args.group,
        args.raw,
        n_min=args.n_min,
        n_max=args.n_max,
    )


if __name__ == "__main__":
    main()

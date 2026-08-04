"""Plot conditional entropy as a function of position n for each hidden_dim run.

Usage:
  python analysis/plot_conditional_entropies.py

Produces results/conditional_entropy_curves.png.
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


def power_law(n, h_inf, c, power):
    return h_inf + c * np.power(n, power)


def power_law_no_offset(n, c, power):
    return c * np.power(n, power)


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
    """Fetch completed runs from a W&B project, optionally filtered by group."""
    api = wandb.Api()
    filters = {"group": group} if group else {}
    runs = api.runs(project, filters=filters)
    return [r for r in runs if r.state == "finished"]


def extract_metrics(
    runs: List[wandb.apis.public.Run], split: str = "combined"
) -> pd.DataFrame:
    """Extract per-position conditional entropies for each run.

    Args:
        runs: List of W&B runs.
        split: Kept for parity with plot_ngrams.py. Ignored for entropy extraction.
    """
    del split
    rows: List[Dict[str, Any]] = []
    for r in runs:
        cfg = r.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        summary = r.summary or {}

        prefixes = ["conditional_entropy/entropy_"]

        for prefix in prefixes:
            for k, v in summary.items():
                if k.startswith(prefix):
                    n = int(k.split("_")[-1])
                    rows.append(
                        {
                            "hidden_dim": int(hidden_dim),
                            "n": n,
                            "entropy": v,
                        }
                    )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["hidden_dim", "n"])


def plot_conditional_entropies(
    df: pd.DataFrame,
    out_path: str,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
    fit_nmin: int = 1,
    fit_nmax: int = 40,
    compare_train: bool = False,
    raw: bool = False,
    plot_hn: bool = False,
    fit_no_offset: bool = False,
) -> None:
    """Plot conditional entropy curves and optional power-law fits.

    Args:
        df: DataFrame with columns [hidden_dim, n, entropy]
        out_path: Output path for the plot
        xlim: X-axis limits
        ylim: Y-axis limits
        fit_nmin: Minimum n for power-law fit window
        fit_nmax: Maximum n for power-law fit window (exclusive)
        compare_train: Kept for parity with plot_ngrams.py
        raw: If True, plot raw H_n without fit subtraction
        plot_hn: If True, plot fitted H_n directly instead of H_n - H_inf
        fit_no_offset: If True, fit H_n = c * n^power (i.e., H_inf fixed to 0)
    """
    if df.empty:
        raise RuntimeError("No completed runs with required metrics found.")

    figsize = (12, 12)

    if compare_train:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base_path = out_path.replace(".png", "")
        unique_hd = sorted(df["hidden_dim"].unique())

        for hidden_dim in unique_hd:
            hd_group = df[df["hidden_dim"] == hidden_dim]
            fig, ax = plt.subplots(figsize=figsize)

            group = hd_group.sort_values("n")
            ns = group["n"].to_numpy(dtype=float)
            entropies = group["entropy"].to_numpy(dtype=float)
            fit_label = "entropy"

            if raw:
                h_inf = 0.0
                power = 0.0
                c = 0.0
                fit_success = False
                y_values = entropies
            else:
                fit_mask = ns >= fit_nmin
                if fit_nmax > 0:
                    fit_mask = fit_mask & (ns < fit_nmax)
                ns_fit, entropies_fit = ns[fit_mask], entropies[fit_mask]

                try:
                    if fit_no_offset:
                        p0 = [max(entropies_fit[0], 1e-8), -0.5]
                        popt, _ = curve_fit(
                            power_law_no_offset,
                            ns_fit,
                            entropies_fit,
                            p0=p0,
                            maxfev=10_000,
                            bounds=([0, -np.inf], [np.inf, 0]),
                        )
                        c, power = popt
                        h_inf = 0.0
                        print(f"hd={hidden_dim}: c={c:.4f}, power={power:.4f}")
                        fit_label = rf"$H_n={c:.3g}\cdot n^{{{power:.3f}}}$"
                    else:
                        p0 = [entropies_fit[-1], entropies_fit[0] - entropies_fit[-1], -0.5]
                        popt, _ = curve_fit(
                            power_law,
                            ns_fit,
                            entropies_fit,
                            p0=p0,
                            maxfev=10_000,
                            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
                        )
                        h_inf, c, power = popt
                        print(
                            f"hd={hidden_dim}: H_inf={h_inf:.4f}, c={c:.4f}, power={power:.4f}"
                        )
                        fit_label = rf"$H_n={c:.3g}\cdot n^{{{power:.3f}}}+{h_inf:.3g}$"
                    fit_success = True
                except RuntimeError as e:
                    print(f"Fit failed for hidden_dim={hidden_dim}: {e}.")
                    h_inf = 0.0 if fit_no_offset else np.min(entropies_fit)
                    fit_label = "fit failed"
                    fit_success = False

                if plot_hn or fit_no_offset:
                    y_values = entropies
                else:
                    y_values = entropies - h_inf

            (line,) = ax.plot(
                ns,
                y_values,
                marker="o",
                markeredgecolor="black",
                linestyle="-",
                label=fit_label,
                color=plt.cm.tab10.colors[0],
                alpha=0.8,
            )
            color = line.get_color()

            if (not raw) and fit_success:
                n_plot_min = max(float(fit_nmin), float(ns[0]))
                n_plot_max = float(ns[-1])
                if fit_nmax > 0:
                    n_plot_max = min(n_plot_max, float(fit_nmax))
                if n_plot_max <= n_plot_min:
                    n_plot_min = float(ns[0])
                    n_plot_max = float(ns[-1])
                ns_fit_dense = np.linspace(n_plot_min, n_plot_max, 200)
                if fit_no_offset:
                    fit_curve = power_law_no_offset(ns_fit_dense, c, power)
                else:
                    fit_curve = power_law(ns_fit_dense, h_inf, c, power)
                if not (plot_hn or fit_no_offset):
                    fit_curve = fit_curve - h_inf
                ax.plot(
                    ns_fit_dense,
                    fit_curve,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )

            ax.set_xlabel("n")
            ylabel = r"$H_n$" if (raw or plot_hn or fit_no_offset) else r"$H_n - H_{\infty}$"
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.10),
                ncol=1,
                borderaxespad=0.0,
                labelspacing=0.25,
                handletextpad=0.4,
                frameon=False,
            )

            fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
            hd_path = f"{base_path}_hd{hidden_dim}.png"
            fig.savefig(hd_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved to {hd_path}")
            _show_image(hd_path)
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=figsize)

        hidden_dims = sorted(df["hidden_dim"].unique())
        norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
        cmap = plt.cm.viridis

        for hidden_dim, hd_group in df.groupby("hidden_dim"):
            group = hd_group.sort_values("n")
            ns = group["n"].to_numpy(dtype=float)
            entropies = group["entropy"].to_numpy(dtype=float)
            fit_label = rf"$d_h={hidden_dim}$"

            if raw:
                h_inf = 0.0
                power = 0.0
                c = 0.0
                fit_success = False
                y_values = entropies
            else:
                fit_mask = ns >= fit_nmin
                if fit_nmax > 0:
                    fit_mask = fit_mask & (ns < fit_nmax)
                ns_fit, entropies_fit = ns[fit_mask], entropies[fit_mask]

                try:
                    if fit_no_offset:
                        p0 = [max(entropies_fit[0], 1e-8), -0.5]
                        popt, _ = curve_fit(
                            power_law_no_offset,
                            ns_fit,
                            entropies_fit,
                            p0=p0,
                            maxfev=10_000,
                            bounds=([0, -np.inf], [np.inf, 0]),
                        )
                        c, power = popt
                        h_inf = 0.0
                        print(f"hd={hidden_dim}: c={c:.4f}, power={power:.4f}")
                        fit_label = (
                            rf"$d_h={hidden_dim}$: "
                            rf"$H_n={c:.3g}\cdot n^{{{power:.3f}}}$"
                        )
                    else:
                        p0 = [entropies_fit[-1], entropies_fit[0] - entropies_fit[-1], -0.5]
                        popt, _ = curve_fit(
                            power_law,
                            ns_fit,
                            entropies_fit,
                            p0=p0,
                            maxfev=10_000,
                            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
                        )
                        h_inf, c, power = popt
                        print(
                            f"hd={hidden_dim}: H_inf={h_inf:.4f}, c={c:.4f}, power={power:.4f}"
                        )
                        fit_label = (
                            rf"$d_h={hidden_dim}$: "
                            rf"$H_n={c:.3g}\cdot n^{{{power:.3f}}}+{h_inf:.3g}$"
                        )
                    fit_success = True
                except RuntimeError as e:
                    print(f"Fit failed for hidden_dim={hidden_dim}: {e}.")
                    h_inf = 0.0 if fit_no_offset else np.min(entropies_fit)
                    fit_label = rf"$d_h={hidden_dim}$: fit failed"
                    fit_success = False

                if plot_hn or fit_no_offset:
                    y_values = entropies
                else:
                    y_values = entropies - h_inf

            (line,) = ax.plot(
                ns,
                y_values,
                marker="o",
                markeredgecolor="black",
                linestyle="-",
                label=fit_label,
                color=cmap(norm(hidden_dim)),
                alpha=0.8,
            )
            color = line.get_color()

            if (not raw) and fit_success:
                n_plot_min = max(float(fit_nmin), float(ns[0]))
                n_plot_max = float(ns[-1])
                if fit_nmax > 0:
                    n_plot_max = min(n_plot_max, float(fit_nmax))
                if n_plot_max <= n_plot_min:
                    n_plot_min = float(ns[0])
                    n_plot_max = float(ns[-1])
                ns_fit_dense = np.linspace(n_plot_min, n_plot_max, 200)
                if fit_no_offset:
                    fit_curve = power_law_no_offset(ns_fit_dense, c, power)
                else:
                    fit_curve = power_law(ns_fit_dense, h_inf, c, power)
                if not (plot_hn or fit_no_offset):
                    fit_curve = fit_curve - h_inf
                ax.plot(
                    ns_fit_dense,
                    fit_curve,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel("n")
        ylabel = r"$H_n$" if (raw or plot_hn or fit_no_offset) else r"$H_n - H_{\infty}$"
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

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
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved to {out_path}")
        _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="W&B group to filter runs by",
    )
    parser.add_argument(
        "--xlim",
        type=int,
        nargs=2,
        default=None,
        help="X-axis range as two integers (e.g., 1 20)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis range as two numbers (e.g., 0.01 1.0)",
    )
    parser.add_argument(
        "--fit-nmin",
        type=int,
        default=1,
        help="Min n to include in the power-law fit (fit uses points with n >= fit_nmin).",
    )
    parser.add_argument(
        "--fit-nmax",
        type=int,
        default=40,
        help=(
            "Max n to include in the power-law fit (fit uses points with n < fit_nmax). "
            "Use <= 0 to fit all available n."
        ),
    )
    parser.add_argument(
        "--compare-train",
        action="store_true",
        help="Generate one figure per hidden_dim (kept for parity with plot_ngrams)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw H_n without fitting or subtracting H_inf",
    )
    parser.add_argument(
        "--plot-hn",
        "--plot-ln",
        dest="plot_hn",
        action="store_true",
        help="Plot fitted H_n (with fit) instead of H_n - H_inf",
    )
    parser.add_argument(
        "--fit-no-offset",
        action="store_true",
        help="Fit H_n = c * n^power instead of H_n = H_inf + c * n^power",
    )
    args = parser.parse_args()
    if args.raw and args.plot_hn:
        raise RuntimeError("--raw and --plot-hn cannot be used together")
    if args.raw and args.fit_no_offset:
        raise RuntimeError("--raw and --fit-no-offset cannot be used together")
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    runs = fetch_runs("scaling", group=args.group)
    df = extract_metrics(runs)

    output_path = "results/conditional_entropy_curves.png"
    if args.compare_train:
        output_path = "results/conditional_entropy_curves_compare.png"

    plot_conditional_entropies(
        df,
        output_path,
        xlim=xlim,
        ylim=ylim,
        fit_nmin=args.fit_nmin,
        fit_nmax=args.fit_nmax,
        compare_train=args.compare_train,
        raw=args.raw,
        plot_hn=args.plot_hn,
        fit_no_offset=args.fit_no_offset,
    )


if __name__ == "__main__":
    main()

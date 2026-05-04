"""Plot n-gram loss as a function of position n for each hidden_dim run.

Usage:
  python analysis/plot_ngrams.py

Produces results/ngram_curves.png.
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


def power_law(n, L_inf, c, power):
    return L_inf + c * np.power(n, power)


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


def _load_cagnetta_data() -> tuple[np.ndarray, np.ndarray]:
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "cagnettaData", "data.csv"
    )
    data = np.loadtxt(data_path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Unexpected Cagnetta data shape: {data.shape}")
    return data[:, 0], data[:, 1]


def extract_metrics(
    runs: List[wandb.apis.public.Run], split: str = "combined"
) -> pd.DataFrame:
    """Extract per-position n-gram losses for each run.

    Args:
        runs: List of W&B runs.
        split: Which split to extract - "combined", "train", or "all".
    """
    rows: List[Dict[str, Any]] = []
    for r in runs:
        cfg = r.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        summary = r.summary or {}

        if split == "all":
            prefixes = ["combined/ngram_", "train_ngram/ngram_"]
        elif split == "train":
            prefixes = ["train_ngram/ngram_"]
        else:
            prefixes = ["combined/ngram_"]

        for prefix in prefixes:
            for k, v in summary.items():
                if k.startswith(prefix):
                    n = int(k.split("_")[-1])
                    dataset = "train" if "train" in prefix else "combined"
                    rows.append(
                        {
                            "hidden_dim": int(hidden_dim),
                            "n": n,
                            "loss": v,
                            "dataset": dataset,
                        }
                    )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["hidden_dim", "n", "dataset"])


def plot_ngrams(
    df: pd.DataFrame,
    out_path: str,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
    fit_nmax: int = 10,
    compare_train: bool = False,
    raw: bool = False,
    plot_ln: bool = False,
    fit_no_offset: bool = False,
    include_cagnetta: bool = False,
) -> None:
    """Plot n-gram loss curves and optional power-law fits.

    Args:
        df: DataFrame with columns [hidden_dim, n, loss, dataset]
        out_path: Output path for the plot
        xlim: X-axis limits
        ylim: Y-axis limits
        fit_nmax: Maximum n for power-law fit
        compare_train: If True, create separate plots per hidden_dim comparing combined vs train
        plot_ln: If True, plot fitted L_n directly instead of L_n - L_inf
        fit_no_offset: If True, fit L_n = c * n^power (i.e., L_inf fixed to 0)
        include_cagnetta: If True, overlay Cagnetta et al. n-gram curve
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

            colors = plt.cm.tab10.colors

            datasets = (
                hd_group["dataset"].unique()
                if "dataset" in hd_group.columns
                else ["combined"]
            )

            for di, dataset in enumerate(datasets):
                group = (
                    hd_group[hd_group["dataset"] == dataset]
                    if "dataset" in hd_group.columns
                    else hd_group
                )
                group = group.sort_values("n")
                ns = group["n"].to_numpy(dtype=float)
                losses = group["loss"].to_numpy(dtype=float)
                fit_label = f"{dataset}"

                if raw:
                    L_inf = 0.0
                    power = 0.0
                    c = 0.0
                    fit_success = False
                    y_values = losses
                else:
                    fit_mask = ns <= fit_nmax
                    ns_fit, losses_fit = ns[fit_mask], losses[fit_mask]

                    try:
                        if fit_no_offset:
                            p0 = [max(losses_fit[0], 1e-8), -0.5]
                            popt, _ = curve_fit(
                                power_law_no_offset,
                                ns_fit,
                                losses_fit,
                                p0=p0,
                                maxfev=10_000,
                                bounds=([0, -np.inf], [np.inf, 0]),
                            )
                            c, power = popt
                            L_inf = 0.0
                            print(
                                f"hd={hidden_dim}, {dataset}: c={c:.4f}, power={power:.4f}"
                            )
                            fit_label = (
                                rf"{dataset}: $L_n={c:.3g}\cdot n^{{{power:.3f}}}$"
                            )
                        else:
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
                                f"hd={hidden_dim}, {dataset}: L_inf={L_inf:.4f}, c={c:.4f}, power={power:.4f}"
                            )
                            fit_label = rf"{dataset}: $L_n={c:.3g}\cdot n^{{{power:.3f}}}+{L_inf:.3g}$"
                        fit_success = True
                    except RuntimeError as e:
                        print(
                            f"Fit failed for hidden_dim={hidden_dim}, {dataset}: {e}."
                        )
                        L_inf = 0.0 if fit_no_offset else np.min(losses_fit)
                        fit_label = f"{dataset}: fit failed"
                        fit_success = False

                    if plot_ln or fit_no_offset:
                        y_values = losses
                    else:
                        y_values = losses - L_inf
                linestyle = "-" if dataset == "combined" else "--"
                marker = "o" if dataset == "combined" else "s"

                (line,) = ax.plot(
                    ns,
                    y_values,
                    marker=marker,
                    markeredgecolor="black",
                    linestyle=linestyle,
                    label=fit_label,
                    color=colors[di % len(colors)],
                    alpha=0.8,
                )
                color = line.get_color()

                if (not raw) and fit_success:
                    ns_fit_dense = np.linspace(1, ns[-1], 200)
                    if fit_no_offset:
                        fit_curve = power_law_no_offset(ns_fit_dense, c, power)
                    else:
                        fit_curve = power_law(ns_fit_dense, L_inf, c, power)
                    if not (plot_ln or fit_no_offset):
                        fit_curve = fit_curve - L_inf
                    ax.plot(
                        ns_fit_dense,
                        fit_curve,
                        color=color,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.8,
                    )

            ax.set_xlabel("n")
            ylabel = (
                r"$L_n$" if (raw or plot_ln or fit_no_offset) else r"$L_n - L_{\infty}$"
            )
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"N-gram loss vs position (d_h = {hidden_dim})")
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
            datasets = (
                hd_group["dataset"].unique()
                if "dataset" in hd_group.columns
                else ["combined"]
            )

            for di, dataset in enumerate(datasets):
                if dataset == "train" and not compare_train:
                    continue

                group = (
                    hd_group[hd_group["dataset"] == dataset]
                    if "dataset" in hd_group.columns
                    else hd_group
                )
                group = group.sort_values("n")
                ns = group["n"].to_numpy(dtype=float)
                losses = group["loss"].to_numpy(dtype=float)
                fit_label = rf"$d_h={hidden_dim}$"

                if raw:
                    L_inf = 0.0
                    power = 0.0
                    c = 0.0
                    fit_success = False
                    y_values = losses
                else:
                    fit_mask = ns <= fit_nmax
                    ns_fit, losses_fit = ns[fit_mask], losses[fit_mask]

                    try:
                        if fit_no_offset:
                            p0 = [max(losses_fit[0], 1e-8), -0.5]
                            popt, _ = curve_fit(
                                power_law_no_offset,
                                ns_fit,
                                losses_fit,
                                p0=p0,
                                maxfev=10_000,
                                bounds=([0, -np.inf], [np.inf, 0]),
                            )
                            c, power = popt
                            L_inf = 0.0
                            print(
                                f"hd={hidden_dim}, {dataset}: c={c:.4f}, power={power:.4f}"
                            )
                            fit_label = (
                                rf"$d_h={hidden_dim}$: "
                                rf"$L_n={c:.3g}\cdot n^{{{power:.3f}}}$"
                            )
                        else:
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
                                f"hd={hidden_dim}, {dataset}: L_inf={L_inf:.4f}, c={c:.4f}, power={power:.4f}"
                            )
                            fit_label = (
                                rf"$d_h={hidden_dim}$: "
                                rf"$L_n={c:.3g}\cdot n^{{{power:.3f}}}+{L_inf:.3g}$"
                            )
                        fit_success = True
                    except RuntimeError as e:
                        print(
                            f"Fit failed for hidden_dim={hidden_dim}, {dataset}: {e}."
                        )
                        L_inf = 0.0 if fit_no_offset else np.min(losses_fit)
                        fit_label = rf"$d_h={hidden_dim}$: fit failed"
                        fit_success = False

                    if plot_ln or fit_no_offset:
                        y_values = losses
                    else:
                        y_values = losses - L_inf

                linestyle = "-" if dataset == "combined" else "--"
                marker = "o" if dataset == "combined" else "s"

                (line,) = ax.plot(
                    ns,
                    y_values,
                    marker=marker,
                    markeredgecolor="black",
                    linestyle=linestyle,
                    label=fit_label,
                    color=cmap(norm(hidden_dim)),
                    alpha=0.8,
                )
                color = line.get_color()

                if (not raw) and fit_success:
                    ns_fit_dense = np.linspace(1, ns[-1], 200)
                    if fit_no_offset:
                        fit_curve = power_law_no_offset(ns_fit_dense, c, power)
                    else:
                        fit_curve = power_law(ns_fit_dense, L_inf, c, power)
                    if not (plot_ln or fit_no_offset):
                        fit_curve = fit_curve - L_inf
                    ax.plot(
                        ns_fit_dense,
                        fit_curve,
                        color=color,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.8,
                    )

        if include_cagnetta:
            ns_cagnetta, losses_cagnetta = _load_cagnetta_data()
            sort_idx = np.argsort(ns_cagnetta)
            ns_cagnetta = ns_cagnetta[sort_idx]
            losses_cagnetta = losses_cagnetta[sort_idx]
            ns_fit = ns_cagnetta[:fit_nmax]
            losses_fit = losses_cagnetta[:fit_nmax]
            fit_success = False
            cagnetta_label = "Cagnetta et al."
            try:
                if fit_no_offset:
                    p0 = [max(losses_fit[0], 1e-8), -0.5]
                    popt, _ = curve_fit(
                        power_law_no_offset,
                        ns_fit,
                        losses_fit,
                        p0=p0,
                        maxfev=10_000,
                        bounds=([0, -np.inf], [np.inf, 0]),
                    )
                    c_c, power_c = popt
                    L_inf_c = 0.0
                else:
                    p0 = [losses_fit[-1], losses_fit[0] - losses_fit[-1], -0.5]
                    popt, _ = curve_fit(
                        power_law,
                        ns_fit,
                        losses_fit,
                        p0=p0,
                        maxfev=10_000,
                        bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
                    )
                    L_inf_c, c_c, power_c = popt
                fit_success = True
                if fit_no_offset:
                    cagnetta_label = rf"Cagnetta et al. ($\nu={power_c:.3f}$)"
                else:
                    cagnetta_label = rf"Cagnetta et al. ($\nu={power_c:.3f}$, $L_\infty={L_inf_c:.3f}$)"
            except RuntimeError:
                L_inf_c = 0.0 if fit_no_offset else np.min(losses_fit)
                c_c = 0.0
                power_c = 0.0

            cagnetta_y = (
                losses_cagnetta
                if (raw or plot_ln or fit_no_offset)
                else losses_cagnetta - L_inf_c
            )
            ax.plot(
                ns_cagnetta,
                cagnetta_y,
                marker="x",
                linestyle="-",
                color="black",
                label=cagnetta_label,
                alpha=0.9,
            )
            if fit_success:
                ns_fit_dense = np.linspace(1, ns_cagnetta[-1], 200)
                if fit_no_offset:
                    fit_curve = power_law_no_offset(ns_fit_dense, c_c, power_c)
                else:
                    fit_curve = power_law(ns_fit_dense, L_inf_c, c_c, power_c)
                if not (raw or plot_ln or fit_no_offset):
                    fit_curve = fit_curve - L_inf_c
                ax.plot(
                    ns_fit_dense,
                    fit_curve,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel("n")
        ylabel = (
            r"$L_n$" if (raw or plot_ln or fit_no_offset) else r"$L_n - L_{\infty}$"
        )
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("N-gram loss vs position")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if (raw or plot_ln or fit_no_offset) and include_cagnetta:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.10),
                ncol=1,
                borderaxespad=0.0,
                labelspacing=0.25,
                handletextpad=0.4,
                frameon=False,
            )
        else:
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
        "--compare-train",
        action="store_true",
        help="Plot both combined (val+test) and training n-grams for comparison",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw L_n without fitting or subtracting L_inf",
    )
    parser.add_argument(
        "--plot-ln",
        action="store_true",
        help="Plot fitted L_n (with fit) instead of L_n - L_inf",
    )
    parser.add_argument(
        "--fit-no-offset",
        action="store_true",
        help="Fit L_n = c * n^power instead of L_n = L_inf + c * n^power",
    )
    parser.add_argument(
        "--include-cagnetta",
        action="store_true",
        help="Overlay Cagnetta et al. n-gram curve",
    )
    args = parser.parse_args()
    if args.raw and args.plot_ln:
        raise RuntimeError("--raw and --plot-ln cannot be used together")
    if args.raw and args.fit_no_offset:
        raise RuntimeError("--raw and --fit-no-offset cannot be used together")
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    runs = fetch_runs("scaling", group=args.group)

    # Determine which splits to fetch based on --compare-train flag
    if args.compare_train:
        split = "all"
    else:
        split = "combined"

    df = extract_metrics(runs, split=split)

    # Determine dataset column presence for filtering
    has_dataset_col = "dataset" in df.columns if not df.empty else False

    # If not comparing train, filter to only combined
    if not args.compare_train and has_dataset_col:
        df = df[df["dataset"] == "combined"]

    output_path = "results/ngram_curves.png"
    if args.compare_train:
        output_path = "results/ngram_curves_compare.png"

    plot_ngrams(
        df,
        output_path,
        xlim=xlim,
        ylim=ylim,
        compare_train=args.compare_train,
        raw=args.raw,
        plot_ln=args.plot_ln,
        fit_no_offset=args.fit_no_offset,
        include_cagnetta=args.include_cagnetta,
    )


if __name__ == "__main__":
    main()

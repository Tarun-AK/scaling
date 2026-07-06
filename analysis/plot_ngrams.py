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
from analysis.plot_group_labels import distinct_group_labels


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


def _load_external_data(source: str) -> tuple[np.ndarray, np.ndarray]:
    source_map = {
        "cagnetta": "cagnetta_ln.csv",
        "kaplan": "kaplan_ln.csv",
        "shengqi": "shengqi_ln.csv",
        "shengi": "shengqi_ln.csv",
    }
    if source not in source_map:
        raise RuntimeError(f"Unsupported external source: {source}")
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "externalData", source_map[source]
    )
    data = np.loadtxt(data_path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"Unexpected external data shape for '{source}': {data.shape}"
        )
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
        group_name = getattr(r, "group", None) or "ungrouped"
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
                            "group": str(group_name),
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
    fit_nmin: int = 1,
    fit_nmax: int = 40,
    compare_train: bool = False,
    raw: bool = False,
    plot_ln: bool = False,
    plot_diff: bool = False,
    fit_no_offset: bool = False,
    include_external: list[str] | None = None,
) -> None:
    """Plot n-gram loss curves and optional power-law fits.

    Args:
        df: DataFrame with columns [hidden_dim, n, loss, dataset]
        out_path: Output path for the plot
        xlim: X-axis limits
        ylim: Y-axis limits
        fit_nmin: Minimum n for power-law fit window
        fit_nmax: Maximum n for power-law fit window (exclusive)
        compare_train: If True, create separate plots per hidden_dim comparing combined vs train
        plot_ln: If True, plot fitted L_n directly instead of L_n - L_inf
        plot_diff: If True, plot raw -diff(L_n)
        fit_no_offset: If True, fit L_n = c * n^power (i.e., L_inf fixed to 0)
        include_external: Optional external curve source(s) to overlay
    """
    if df.empty:
        raise RuntimeError("No completed runs with required metrics found.")

    figsize = (12, 12)
    unique_groups = (
        sorted(df["group"].dropna().astype(str).unique().tolist())
        if "group" in df.columns
        else []
    )
    group_labels = distinct_group_labels(unique_groups) if len(unique_groups) > 1 else {}
    show_group_name = "group" in df.columns and df["group"].nunique() > 1

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
                group_name = (
                    str(group["group"].iloc[0])
                    if ("group" in group.columns and not group.empty)
                    else None
                )
                group_label = group_labels.get(group_name, group_name) if group_name else None
                ns = group["n"].to_numpy(dtype=float)
                losses = group["loss"].to_numpy(dtype=float)
                fit_label = (
                    f"{dataset} [{group_label}]"
                    if show_group_name and group_label is not None
                    else f"{dataset}"
                )

                if plot_diff:
                    ns_diff = ns[1:]
                    losses_diff = -np.diff(losses)
                    valid = (
                        np.isfinite(ns_diff)
                        & np.isfinite(losses_diff)
                        & (losses_diff > 0)
                    )
                    ns_plot = ns_diff[valid]
                    y_values = losses_diff[valid]
                    if ns_plot.size < 2:
                        print(
                            f"Skipping hidden_dim={hidden_dim}, {dataset}: insufficient positive -diff values"
                        )
                        continue
                    fit_label = (
                        f"{dataset} [{group_label}]"
                        if show_group_name and group_label is not None
                        else f"{dataset}"
                    )
                    fit_success = False
                    fit_points = min(5, int(ns_plot.size))
                    if fit_points >= 2:
                        ns_fit = ns_plot[:fit_points]
                        losses_fit = y_values[:fit_points]
                        try:
                            p0 = [max(float(losses_fit[0]), 1e-8), -1.0]
                            popt, _ = curve_fit(
                                power_law_no_offset,
                                ns_fit,
                                losses_fit,
                                p0=p0,
                                maxfev=10_000,
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                            )
                            c, power = popt
                            fit_success = True
                            fit_label = (
                                rf"{dataset}: "
                                rf"$-\Delta L_n={c:.3g}\cdot n^{{{power:.3f}}}$ "
                                rf"(fit first {fit_points})"
                            )
                            print(
                                f"hd={hidden_dim}, {dataset}: diff fit first {fit_points} "
                                f"c={c:.4f}, power={power:.4f}"
                            )
                        except (RuntimeError, ValueError) as e:
                            print(
                                f"Diff fit failed for hidden_dim={hidden_dim}, {dataset}: {e}."
                            )

                    marker = "o" if dataset == "combined" else "s"
                    color = colors[di % len(colors)]
                    ax.scatter(
                        ns_plot,
                        y_values,
                        marker=marker,
                        edgecolors="black",
                        label=fit_label,
                        color=color,
                        alpha=0.8,
                    )
                    if fit_success:
                        ns_fit_dense = np.linspace(
                            float(ns_plot[0]), float(ns_plot[-1]), 200
                        )
                        fit_curve = power_law_no_offset(ns_fit_dense, c, power)
                        ax.plot(
                            ns_fit_dense,
                            fit_curve,
                            color="red",
                            linestyle="-",
                            linewidth=2.0,
                            alpha=0.8,
                        )
                    continue

                if raw:
                    L_inf = 0.0
                    power = 0.0
                    c = 0.0
                    fit_success = False
                    y_values = losses
                else:
                    fit_mask = ns >= fit_nmin
                    if fit_nmax > 0:
                        fit_mask = fit_mask & (ns < fit_nmax)
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
                    except (RuntimeError, ValueError) as e:
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
                marker = "o" if dataset == "combined" else "s"
                color = colors[di % len(colors)]
                ax.scatter(
                    ns,
                    y_values,
                    marker=marker,
                    edgecolors="black",
                    label=fit_label,
                    color=color,
                    alpha=0.8,
                )

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
                        fit_curve = power_law(ns_fit_dense, L_inf, c, power)
                    if not (plot_ln or fit_no_offset):
                        fit_curve = fit_curve - L_inf
                    ax.plot(
                        ns_fit_dense,
                        fit_curve,
                        color="red",
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.8,
                    )

            ax.set_xlabel("n")
            if plot_diff:
                ylabel = r"$-\Delta L_n$"
            else:
                ylabel = (
                    r"$L_n$"
                    if (raw or plot_ln or fit_no_offset)
                    else r"$L_n - L_{\infty}$"
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
                group_name = (
                    str(group["group"].iloc[0])
                    if ("group" in group.columns and not group.empty)
                    else None
                )
                group_label = group_labels.get(group_name, group_name) if group_name else None
                ns = group["n"].to_numpy(dtype=float)
                losses = group["loss"].to_numpy(dtype=float)
                fit_label = rf"$d_h={hidden_dim}$" + (
                    rf" [{group_label}]" if group_label is not None else ""
                )

                if plot_diff:
                    ns_diff = ns[1:]
                    losses_diff = -np.diff(losses)
                    valid = (
                        np.isfinite(ns_diff)
                        & np.isfinite(losses_diff)
                        & (losses_diff > 0)
                    )
                    ns_plot = ns_diff[valid]
                    y_values = losses_diff[valid]
                    if ns_plot.size < 2:
                        print(
                            f"Skipping hidden_dim={hidden_dim}, {dataset}: insufficient positive -diff values"
                        )
                        continue
                    fit_label = rf"$d_h={hidden_dim}$" + (
                        rf" [{group_label}]" if group_label is not None else ""
                    )
                    fit_success = False
                    fit_points = min(5, int(ns_plot.size))
                    if fit_points >= 2:
                        ns_fit = ns_plot[:fit_points]
                        losses_fit = y_values[:fit_points]
                        try:
                            p0 = [max(float(losses_fit[0]), 1e-8), -1.0]
                            popt, _ = curve_fit(
                                power_law_no_offset,
                                ns_fit,
                                losses_fit,
                                p0=p0,
                                maxfev=10_000,
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                            )
                            c, power = popt
                            fit_success = True
                            fit_label = (
                                rf"$d_h={hidden_dim}$"
                                + (rf" [{group_label}]" if group_label is not None else "")
                                + ": "
                                rf"$-\Delta L_n={c:.3g}\cdot n^{{{power:.3f}}}$ "
                                rf"(fit first {fit_points})"
                            )
                            print(
                                f"hd={hidden_dim}, {dataset}: diff fit first {fit_points} "
                                f"c={c:.4f}, power={power:.4f}"
                            )
                        except (RuntimeError, ValueError) as e:
                            print(
                                f"Diff fit failed for hidden_dim={hidden_dim}, {dataset}: {e}."
                            )

                    marker = "o" if dataset == "combined" else "s"
                    color = cmap(norm(hidden_dim))
                    ax.scatter(
                        ns_plot,
                        y_values,
                        marker=marker,
                        edgecolors="black",
                        label=fit_label,
                        color=color,
                        alpha=0.8,
                    )
                    if fit_success:
                        ns_fit_dense = np.linspace(
                            float(ns_plot[0]), float(ns_plot[-1]), 200
                        )
                        fit_curve = power_law_no_offset(ns_fit_dense, c, power)
                        ax.plot(
                            ns_fit_dense,
                            fit_curve,
                            color="red",
                            linestyle="-",
                            linewidth=2.0,
                            alpha=0.8,
                        )
                    continue

                if raw:
                    L_inf = 0.0
                    power = 0.0
                    c = 0.0
                    fit_success = False
                    y_values = losses
                else:
                    fit_mask = ns >= fit_nmin
                    if fit_nmax > 0:
                        fit_mask = fit_mask & (ns < fit_nmax)
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
                                rf"$d_h={hidden_dim}$"
                                + (rf" [{group_label}]" if group_label is not None else "")
                                + ": "
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
                                rf"$d_h={hidden_dim}$"
                                + (rf" [{group_label}]" if group_label is not None else "")
                                + ": "
                                rf"$L_n={c:.3g}\cdot n^{{{power:.3f}}}+{L_inf:.3g}$"
                            )
                        fit_success = True
                    except (RuntimeError, ValueError) as e:
                        print(
                            f"Fit failed for hidden_dim={hidden_dim}, {dataset}: {e}."
                        )
                        L_inf = 0.0 if fit_no_offset else np.min(losses_fit)
                        fit_label = (
                                rf"$d_h={hidden_dim}$"
                                + (rf" [{group_label}]" if group_label is not None else "")
                                + ": fit failed"
                            )
                        fit_success = False

                    if plot_ln or fit_no_offset:
                        y_values = losses
                    else:
                        y_values = losses - L_inf

                marker = "o" if dataset == "combined" else "s"
                color = cmap(norm(hidden_dim))
                ax.scatter(
                    ns,
                    y_values,
                    marker=marker,
                    edgecolors="black",
                    label=fit_label,
                    color=color,
                    alpha=0.8,
                )

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
                        fit_curve = power_law(ns_fit_dense, L_inf, c, power)
                    if not (plot_ln or fit_no_offset):
                        fit_curve = fit_curve - L_inf
                    ax.plot(
                        ns_fit_dense,
                        fit_curve,
                        color="red",
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.8,
                    )

        if include_external:
            ext_colors = ["black", "dimgray", "slategray", "gray"]
            ext_markers = ["x", "+", "1", "2"]
            for ext_idx, external_source in enumerate(include_external):
                ns_external, losses_external = _load_external_data(external_source)
                external_label_prefix = {
                    "cagnetta": "Cagnetta et al.",
                    "kaplan": "Kaplan et al.",
                    "shengqi": "Shengqi et al.",
                    "shengi": "Shengqi et al.",
                }[external_source]
                sort_idx = np.argsort(ns_external)
                ns_external = ns_external[sort_idx]
                losses_external = losses_external[sort_idx]
                fit_mask_external = ns_external >= fit_nmin
                if fit_nmax > 0:
                    fit_mask_external = fit_mask_external & (ns_external < fit_nmax)
                ns_fit = ns_external[fit_mask_external]
                losses_fit = losses_external[fit_mask_external]
                L_inf_c = 0.0
                fit_success = False
                external_label = external_label_prefix
                ns_external_plot = ns_external
                external_y = losses_external
                c_c = 0.0
                power_c = 0.0
                try:
                    if plot_diff:
                        ns_diff = ns_external[1:]
                        losses_diff = -np.diff(losses_external)
                        valid = (
                            np.isfinite(ns_diff)
                            & np.isfinite(losses_diff)
                            & (losses_diff > 0)
                        )
                        ns_external_plot = ns_diff[valid]
                        external_y = losses_diff[valid]
                        fit_success = False
                        external_label = external_label_prefix
                        fit_points = min(5, int(ns_external_plot.size))
                        if fit_points >= 2:
                            ns_fit = ns_external_plot[:fit_points]
                            losses_fit = external_y[:fit_points]
                            try:
                                if ns_fit.size == 0 or losses_fit.size == 0:
                                    raise RuntimeError("No external diff points in fit window")
                                p0 = [max(float(losses_fit[0]), 1e-8), -1.0]
                                popt, _ = curve_fit(
                                    power_law_no_offset,
                                    ns_fit,
                                    losses_fit,
                                    p0=p0,
                                    maxfev=10_000,
                                    bounds=([0, -np.inf], [np.inf, np.inf]),
                                )
                                c_c, power_c = popt
                                fit_success = True
                                external_label = (
                                    rf"{external_label_prefix}: "
                                    rf"$-\Delta L_n={c_c:.3g}\cdot n^{{{power_c:.3f}}}$ "
                                    rf"(fit first {fit_points})"
                                )
                            except (RuntimeError, ValueError):
                                fit_success = False
                    elif fit_no_offset:
                        if ns_fit.size == 0 or losses_fit.size == 0:
                            raise RuntimeError("No external points in fit window")
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
                        if ns_fit.size == 0 or losses_fit.size == 0:
                            raise RuntimeError("No external points in fit window")
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
                    if not plot_diff:
                        fit_success = True
                        if fit_no_offset:
                            external_label = (
                                rf"{external_label_prefix}: "
                                rf"$L_n={c_c:.3g}\cdot n^{{{power_c:.3f}}}$"
                            )
                        else:
                            external_label = (
                                rf"{external_label_prefix}: "
                                rf"$L_n={c_c:.3g}\cdot n^{{{power_c:.3f}}}+{L_inf_c:.3g}$"
                            )
                except (RuntimeError, ValueError):
                    if losses_fit.size > 0:
                        L_inf_c = (
                            0.0 if (fit_no_offset or plot_diff) else np.min(losses_fit)
                        )
                    else:
                        L_inf_c = 0.0
                    fit_success = False

                if not plot_diff:
                    external_y = (
                        losses_external
                        if (raw or plot_ln or fit_no_offset)
                        else losses_external - L_inf_c
                    )
                    ns_external_plot = ns_external
                if ns_external_plot.size == 0 or external_y.size == 0:
                    continue
                ext_color = ext_colors[ext_idx % len(ext_colors)]
                ext_marker = ext_markers[ext_idx % len(ext_markers)]
                ax.scatter(
                    ns_external_plot,
                    external_y,
                    marker=ext_marker,
                    color=ext_color,
                    label=external_label,
                    alpha=0.9,
                )
                if fit_success:
                    n_plot_min = max(float(fit_nmin), float(ns_external_plot[0]))
                    n_plot_max = float(ns_external_plot[-1])
                    if fit_nmax > 0:
                        n_plot_max = min(n_plot_max, float(fit_nmax))
                    if n_plot_max <= n_plot_min:
                        n_plot_min = float(ns_external_plot[0])
                        n_plot_max = float(ns_external_plot[-1])
                    ns_fit_dense = np.linspace(n_plot_min, n_plot_max, 200)
                    if plot_diff:
                        fit_curve = power_law_no_offset(ns_fit_dense, c_c, power_c)
                    elif fit_no_offset:
                        fit_curve = power_law_no_offset(ns_fit_dense, c_c, power_c)
                    else:
                        fit_curve = power_law(ns_fit_dense, L_inf_c, c_c, power_c)
                    if not (raw or plot_ln or fit_no_offset or plot_diff):
                        fit_curve = fit_curve - L_inf_c
                    ax.plot(
                        ns_fit_dense,
                        fit_curve,
                        color="red",
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.8,
                    )

        ax.set_xlabel("n")
        if plot_diff:
            ylabel = r"$-\Delta L_n$"
        else:
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

        if (raw or plot_ln or fit_no_offset or plot_diff) and include_external:
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
        nargs="+",
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
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
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
        help="Plot fitted L_n directly instead of L_n - L_inf",
    )
    parser.add_argument(
        "--plot-diff",
        action="store_true",
        help="Plot raw -diff(L_n) with no fit",
    )
    parser.add_argument(
        "--fit-no-offset",
        action="store_true",
        help="Fit L_n = c * n^power instead of L_n = L_inf + c * n^power",
    )
    parser.add_argument(
        "--include-external",
        type=str,
        nargs="+",
        choices=["cagnetta", "kaplan", "shengqi", "shengi"],
        default=None,
        help="Overlay one or more external n-gram curves",
    )
    args = parser.parse_args()
    if args.raw and args.plot_ln:
        raise RuntimeError("--raw and --plot-ln cannot be used together")
    if args.plot_ln and args.plot_diff:
        raise RuntimeError("--plot-ln and --plot-diff cannot be used together")
    if args.raw and args.plot_diff:
        raise RuntimeError("--raw and --plot-diff cannot be used together")
    if args.raw and args.fit_no_offset:
        raise RuntimeError("--raw and --fit-no-offset cannot be used together")
    if args.fit_no_offset and args.plot_diff:
        raise RuntimeError("--fit-no-offset and --plot-diff cannot be used together")
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    if args.group:
        groups = list(dict.fromkeys(args.group))
        runs: list[wandb.apis.public.Run] = []
        for group_name in groups:
            runs.extend(fetch_runs("scaling", group=group_name))
    else:
        groups = []
        runs = fetch_runs("scaling", group=None)

    # Determine which splits to fetch based on --compare-train flag
    if args.compare_train:
        split = "all"
    else:
        split = "combined"

    df = extract_metrics(runs, split=split)

    if args.hidden_dim is not None and not df.empty:
        hidden_dims = set(args.hidden_dim)
        df = df[df["hidden_dim"].isin(hidden_dims)]
        if df.empty:
            raise RuntimeError(
                "No n-gram metrics found for requested hidden_dim filter: "
                f"{sorted(hidden_dims)}"
            )

    # Determine dataset column presence for filtering
    has_dataset_col = "dataset" in df.columns if not df.empty else False

    # If not comparing train, filter to only combined
    if not args.compare_train and has_dataset_col:
        df = df[df["dataset"] == "combined"]

    output_path = "results/ngram_curves.png"
    if args.compare_train:
        output_path = "results/ngram_curves_compare.png"
    if groups:
        output_path = output_path.replace(
            ".png",
            f"_{'_'.join(groups)}.png",
        )

    include_external = None
    if args.include_external:
        include_external = list(dict.fromkeys(args.include_external))

    plot_ngrams(
        df,
        output_path,
        xlim=xlim,
        ylim=ylim,
        fit_nmin=args.fit_nmin,
        fit_nmax=args.fit_nmax,
        compare_train=args.compare_train,
        raw=args.raw,
        plot_ln=args.plot_ln,
        plot_diff=args.plot_diff,
        fit_no_offset=args.fit_no_offset,
        include_external=include_external,
    )


if __name__ == "__main__":
    main()

"""Plot final train vs test loss as a function of hidden dimension.

Usage:
    python analysis/train_vs_val.py

Produces results/overfitting.png.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import termios
import tty
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import wandb

WANDB_PROJECT = "tarunadvaith-/scaling"
TRAIN_AVG_LAST_N = 100
TEST_FIT_OFFSET = 2.25
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


def fetch_runs(
    project: str,
    group: str | None = None,
) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    filters = {"group": group} if group else {}
    runs = api.runs(project, filters=filters)
    return [r for r in runs if r.state in ("finished", "running")]


def extract_final_losses(
    runs: List[wandb.apis.public.Run],
    group: str | None = None,
) -> pd.DataFrame:
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

        test_loss_col = "test/loss"
        if test_loss_col in history.columns:
            test_series = history[test_loss_col].dropna()
            if len(test_series) == 0:
                continue
            final_test = float(test_series.iloc[-1])
        else:
            test_cols = [c for c in history.columns if c.startswith("test/ngram_")]
            if not test_cols:
                continue
            last_row = history[test_cols].dropna(how="all").iloc[-1]
            final_test = float(last_row.mean())

        rows.append(
            {
                "hidden_dim": int(hidden_dim),
                "train_loss": final_train,
                "test_loss": final_test,
                "group": group,
            }
        )

    return pd.DataFrame(rows).sort_values("hidden_dim")


def plot_overfitting(
    df: pd.DataFrame,
    out_path: str,
    fit: bool = True,
) -> None:
    if df.empty:
        raise RuntimeError("No data found.")

    plt.figure(figsize=(10, 7))
    train_color = "tab:blue"
    test_color = "tab:orange"
    markers = ["o", "^", "D", "v", "P", "X"]

    if "group" in df.columns and df["group"].notna().any():
        groups = [g for g in df["group"].unique() if g is not None]
    else:
        groups = [None]
    show_group_label = len(groups) > 1

    for gi, group in enumerate(groups):
        marker = markers[gi % len(markers)]
        group_df = df if group is None else df[df["group"] == group]
        label_suffix = f" ({group})" if group and show_group_label else ""

        plt.scatter(
            group_df["hidden_dim"],
            group_df["train_loss"],
            label=f"train loss (final){label_suffix}",
            edgecolor="black",
            color=train_color,
            marker=marker,
            alpha=0.8,
        )
        plt.scatter(
            group_df["hidden_dim"],
            group_df["test_loss"],
            label=f"test loss (final){label_suffix}",
            edgecolor="black",
            color=test_color,
            marker=marker,
            alpha=0.6,
        )

        if fit:
            x = group_df["hidden_dim"].to_numpy(dtype=float)
            y = group_df["test_loss"].to_numpy(dtype=float)
            fit_mask = (x > 0) & (y > 0)
            if np.count_nonzero(fit_mask) >= 2:
                x_fit_data = x[fit_mask]
                y_fit_data = y[fit_mask]

                try:
                    def power_law_with_fixed_offset(dh, c, power):
                        return TEST_FIT_OFFSET + c * np.power(dh, power)

                    y_shifted = y_fit_data - TEST_FIT_OFFSET
                    valid_shifted = y_shifted > 0
                    if np.count_nonzero(valid_shifted) < 2:
                        continue
                    x_fit_data = x_fit_data[valid_shifted]
                    y_fit_data = y_fit_data[valid_shifted]

                    popt, pcov = curve_fit(
                        power_law_with_fixed_offset,
                        x_fit_data,
                        y_fit_data,
                        p0=[float(np.max(y_fit_data) - TEST_FIT_OFFSET), -0.5],
                        maxfev=10_000,
                        bounds=([0, -np.inf], [np.inf, 0]),
                    )
                    c, power = popt
                    fit_fn = power_law_with_fixed_offset
                    power_idx = 1
                    fit_name = rf"test fit ($L_\infty={TEST_FIT_OFFSET:.2f}$ fixed)"

                    power_std = np.nan
                    if (
                        np.all(np.isfinite(pcov))
                        and pcov.ndim == 2
                        and pcov.shape[0] > power_idx
                        and pcov.shape[1] > power_idx
                        and float(pcov[power_idx, power_idx]) >= 0.0
                    ):
                        power_std = float(np.sqrt(pcov[power_idx, power_idx]))
                    if np.isfinite(power_std):
                        fit_label = (
                            rf"{fit_name} (power={power:.3f}"
                            rf"$\pm${power_std:.3f}){label_suffix}"
                        )
                    else:
                        fit_label = rf"{fit_name} (power={power:.3f}){label_suffix}"
                    x_fit = np.linspace(x_fit_data.min(), x_fit_data.max(), 200)
                    y_fit = fit_fn(x_fit, *popt)
                    plt.plot(
                        x_fit,
                        y_fit,
                        linestyle=":",
                        color=test_color,
                        alpha=0.8,
                        label=fit_label,
                    )
                    if np.all(np.isfinite(pcov)):
                        x_power = np.power(x_fit, power)
                        jac = np.column_stack([x_power, c * x_power * np.log(x_fit)])
                        var = np.einsum("ij,jk,ik->i", jac, pcov, jac)
                        sigma = np.sqrt(np.maximum(var, 0.0))
                        y_lower = np.maximum(y_fit - sigma, np.finfo(float).tiny)
                        y_upper = np.maximum(y_fit + sigma, np.finfo(float).tiny)
                        plt.fill_between(
                            x_fit,
                            y_lower,
                            y_upper,
                            color=test_color,
                            alpha=0.15,
                        )
                except RuntimeError:
                    pass
    if groups and any(g is not None for g in groups):
        title_groups = ", ".join([g for g in groups if g is not None])
        plt.title(f"{title_groups}")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(r"$d_h$")
    plt.ylabel("L")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        action="append",
        default=None,
        help="W&B group to filter runs by (repeatable)",
    )
    parser.add_argument(
        "--fit",
        dest="fit",
        action="store_true",
        help="Fit a power law to test loss",
    )
    parser.add_argument(
        "--no-fit",
        dest="fit",
        action="store_false",
        help="Disable power-law fit",
    )
    parser.add_argument(
        "--fit-without-offset",
        action="store_true",
        help="Deprecated; ignored (fit uses fixed offset L_inf=2.25)",
    )
    parser.set_defaults(fit=True)
    args = parser.parse_args()

    if args.group:
        dfs = []
        for group in args.group:
            runs = fetch_runs(WANDB_PROJECT, group=group)
            dfs.append(extract_final_losses(runs, group=group))
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        runs = fetch_runs(WANDB_PROJECT, group=None)
        df = extract_final_losses(runs, group=None)
    plot_overfitting(
        df,
        "results/overfitting.png",
        fit=args.fit,
    )


if __name__ == "__main__":
    main()

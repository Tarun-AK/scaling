"""Plot L_infinity and test loss vs hidden_dim."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import termios
import tty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import wandb

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


def power_law(n, l_inf, c, power):
    return l_inf + c * np.power(n, power)


def fetch_runs(project: str, group: str | None = None) -> list[wandb.apis.public.Run]:
    api = wandb.Api()
    filters = {"group": group} if group else {}
    runs = api.runs(project, filters=filters)
    return [r for r in runs if r.state == "finished"]


def _extract_ngram_index(metric_key: str) -> int | None:
    name = metric_key.split("/")[-1]
    if name.startswith("ngram_"):
        return int(name.split("_")[-1])
    if name.startswith("n_gram_"):
        return int(name.split("_")[-1])
    return None


def _combined_ngram_keys(keys: list[str]) -> list[str]:
    return [
        key
        for key in keys
        if key.startswith("combined/ngram_") or key.startswith("combined/n_gram_")
    ]


def extract_l_ngrams(runs: list[wandb.apis.public.Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        hidden_dim = run.config.get("hidden_dim")
        if hidden_dim is None:
            continue
        hidden_dim = int(hidden_dim)

        summary = run.summary or {}
        combined_keys = _combined_ngram_keys(list(summary.keys()))
        if combined_keys:
            for key in combined_keys:
                n = _extract_ngram_index(key)
                value = summary.get(key)
                if n is None or not pd.notna(value):
                    continue
                rows.append(
                    {
                        "hidden_dim": hidden_dim,
                        "n": n,
                        "loss": float(value),
                    }
                )
            continue

        history_cols = list(run.history(samples=1).columns)
        combined_keys = _combined_ngram_keys(history_cols)
        if not combined_keys:
            continue
        history = run.history(keys=combined_keys, samples=10000)
        if history.empty:
            continue
        valid = history[combined_keys].dropna(how="all")
        if valid.empty:
            continue
        last_row = valid.iloc[-1]
        for key in combined_keys:
            n = _extract_ngram_index(key)
            value = last_row.get(key)
            if n is None or not pd.notna(value):
                continue
            rows.append(
                {
                    "hidden_dim": hidden_dim,
                    "n": n,
                    "loss": float(value),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["hidden_dim", "n", "loss"])
    return pd.DataFrame(rows).sort_values(["hidden_dim", "n"])


def fit_l_infinity(df: pd.DataFrame, fit_nmax: int = 50) -> pd.DataFrame:
    rows = []
    for hidden_dim, group in df.groupby("hidden_dim"):
        group = group.sort_values("n")
        ns = group["n"].to_numpy(dtype=float)
        losses = group["loss"].to_numpy(dtype=float)
        fit_mask = ns <= fit_nmax
        ns_fit = ns[fit_mask]
        losses_fit = losses[fit_mask]
        if len(ns_fit) == 0:
            continue
        p0 = [losses_fit[-1], losses_fit[0] - losses_fit[-1], -0.5]
        try:
            popt, _ = curve_fit(
                power_law,
                ns_fit,
                losses_fit,
                p0=p0,
                maxfev=10_000,
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
            )
            l_inf = float(popt[0])
        except RuntimeError:
            l_inf = float(np.min(losses_fit))
        rows.append({"hidden_dim": int(hidden_dim), "l_inf": l_inf})
    if not rows:
        return pd.DataFrame(columns=["hidden_dim", "l_inf"])
    return pd.DataFrame(rows).sort_values("hidden_dim")


def extract_final_losses(runs: list[wandb.apis.public.Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        hidden_dim = run.config.get("hidden_dim")
        if hidden_dim is None:
            continue
        history = run.history()
        if history is None or history.empty:
            continue

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
                "test_loss": final_test,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["hidden_dim", "test_loss"])
    return (
        pd.DataFrame(rows)
        .groupby("hidden_dim", as_index=False)[["test_loss"]]
        .mean()
        .sort_values("hidden_dim")
    )


def plot_l_infinity(
    l_inf_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: str,
    title: str | None,
) -> None:
    if l_inf_df.empty and test_df.empty:
        raise RuntimeError("No data found.")
    plt.figure(figsize=(8, 6))

    def _plot_with_fit(
        x: np.ndarray,
        y: np.ndarray,
        *,
        marker: str,
        label_base: str,
        asymptote_label: str,
        color: str | None = None,
    ) -> float | None:
        fit_mask = (x > 0) & (y > 0)
        nu = None
        asymptote = None
        x_fit = None
        y_fit = None
        if np.count_nonzero(fit_mask) >= 2:
            x_fit = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 200)

            def _model(x_in, offset, coef, power):
                return offset + coef * np.power(x_in, power)

            offset0 = float(np.min(y[fit_mask]))
            coef0 = float(np.max(y[fit_mask]) - offset0)
            p0 = [offset0, coef0, -0.5]
            try:
                popt, _ = curve_fit(
                    _model,
                    x[fit_mask],
                    y[fit_mask],
                    p0=p0,
                    maxfev=10_000,
                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0]),
                )
                asymptote = float(popt[0])
                nu = -float(popt[2])
                y_fit = _model(x_fit, *popt)
            except RuntimeError:
                y_fit = None

        fit_terms = []
        if asymptote is not None:
            fit_terms.append(rf"{asymptote_label}={asymptote:.3g}")
        if nu is not None:
            fit_terms.append(rf"$\nu$={nu:.3f}")
        label = f"{label_base} ({', '.join(fit_terms)})" if fit_terms else label_base
        kwargs = {
            "marker": marker,
            "markeredgecolor": "black",
            "alpha": 0.8,
            "label": label,
        }
        if color is not None:
            kwargs["color"] = color
        (line,) = plt.plot(x, y, **kwargs)
        if nu is not None and x_fit is not None and y_fit is not None:
            plt.plot(
                x_fit,
                y_fit,
                linestyle=":",
                alpha=0.8,
                color=line.get_color(),
            )
        return nu

    nu_l_inf = None
    if not l_inf_df.empty:
        nu_l_inf = _plot_with_fit(
            l_inf_df["hidden_dim"].to_numpy(dtype=float),
            l_inf_df["l_inf"].to_numpy(dtype=float),
            marker="o",
            label_base=r"$L_{\infty}(d_h)$",
            asymptote_label=r"$L_{\infty}(\infty)$",
        )

    has_test = not test_df.empty
    if has_test:
        _plot_with_fit(
            test_df["hidden_dim"].to_numpy(dtype=float),
            test_df["test_loss"].to_numpy(dtype=float),
            marker="s",
            label_base=r"$L(d_h)$",
            asymptote_label=r"$L(\infty)$",
            color="tab:orange",
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(r"$d_h$")
    plt.ylabel("")
    if title:
        plt.title(title)
    if (nu_l_inf is not None) or has_test:
        plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/L_infinity.png")
    parser.add_argument("--max-hidden-dim", type=int, default=None)
    args = parser.parse_args()

    runs = fetch_runs("tarunadvaith-/scaling", group=args.group)
    l_n_df = extract_l_ngrams(runs)
    if args.max_hidden_dim is not None:
        l_n_df = l_n_df[l_n_df["hidden_dim"] <= args.max_hidden_dim]
    l_inf_df = fit_l_infinity(l_n_df)
    test_df = extract_final_losses(runs)
    if args.max_hidden_dim is not None:
        test_df = test_df[test_df["hidden_dim"] <= args.max_hidden_dim]
    title = args.group if args.group else None
    plot_l_infinity(l_inf_df, test_df, args.output, title)


if __name__ == "__main__":
    main()

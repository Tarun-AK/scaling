"""Plot L_{n*(d_h)}(d_h->infinity) and L(d_h) vs hidden_dim."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import termios
import tty

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import wandb
from analysis.plot_bipartite_mi import _filter_runs_by_hidden_dim, _resolve_group_runs
from analysis.plot_n_star import (
    _extract_combined_ngram_losses,
    _interpolated_n_for_target_value,
    _load_external_l_reference,
    _mean_last_n_values,
)

plt.style.use("~/plotStyle.mplstyle")


def _power_law(x: np.ndarray, coef: float, power: float) -> np.ndarray:
    return coef * np.power(x, power)


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Fit y = coef * x^power in log-log space."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 2:
        return None

    logx = np.log(x_fit)
    logy = np.log(y_fit)
    try:
        (power, log_coef), cov = np.polyfit(logx, logy, deg=1, cov=True)
    except Exception:
        return None

    coef = float(np.exp(log_coef))
    power = float(power)
    power_stderr = float(np.sqrt(cov[0, 0])) if cov.shape == (2, 2) else float("nan")
    return coef, power, power_stderr


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


def _mean_combined_ngram_loss(series: dict[int, float]) -> float:
    if not series:
        raise RuntimeError("Cannot compute L(d_h) from empty combined/ngram series")
    values = np.array([float(v) for v in series.values()], dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise RuntimeError(
            "Cannot compute L(d_h) from non-finite combined/ngram values"
        )
    return float(np.mean(finite_values))


def _plot_curves(rows: list[dict[str, float]], out_path: str, title: str) -> None:
    if not rows:
        raise RuntimeError("No rows to plot")

    hidden_dims = np.array([row["hidden_dim"] for row in rows], dtype=float)
    l_at_nstar_inf = np.array([row["l_nstar_inf"] for row in rows], dtype=float)
    l_test_dh = np.array([row["l_test_dh"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    (line_nstar,) = ax.plot(
        hidden_dims,
        l_at_nstar_inf,
        marker="o",
        linestyle="-",
        label=r"$L_{n^*_L(d_h)}(d_h \to \infty)$",
    )
    fit_nstar = _fit_power_law(hidden_dims, l_at_nstar_inf)
    if fit_nstar is not None:
        coef, power, power_stderr = fit_nstar
        x_fit = np.geomspace(np.min(hidden_dims), np.max(hidden_dims), 200)
        y_fit = _power_law(x_fit, coef, power)
        if np.isfinite(power_stderr):
            fit_label = (
                rf"fit $L_{{n^*_L}}$: $L={coef:.3g}d_h^{{{power:.3f}}}$, "
                rf"$b={power:.3f}\pm{power_stderr:.3f}$"
            )
        else:
            fit_label = rf"fit $L_{{n^*_L}}$: $L={coef:.3g}d_h^{{{power:.3f}}}$"
        ax.plot(
            x_fit,
            y_fit,
            color=line_nstar.get_color(),
            linestyle=":",
            linewidth=1.5,
            label=fit_label,
        )

    (line_test,) = ax.plot(
        hidden_dims,
        l_test_dh,
        marker="d",
        linestyle="-.",
        label=r"$L_{test}(d_h)$",
    )
    fit_test = _fit_power_law(hidden_dims, l_test_dh)
    if fit_test is not None:
        coef, power, power_stderr = fit_test
        x_fit = np.geomspace(np.min(hidden_dims), np.max(hidden_dims), 200)
        y_fit = _power_law(x_fit, coef, power)
        if np.isfinite(power_stderr):
            fit_label = (
                rf"fit $L_{{test}}$: $L={coef:.3g}d_h^{{{power:.3f}}}$, "
                rf"$b={power:.3f}\pm{power_stderr:.3f}$"
            )
        else:
            fit_label = rf"fit $L_{{test}}$: $L={coef:.3g}d_h^{{{power:.3f}}}$"
        ax.plot(
            x_fit,
            y_fit,
            color=line_test.get_color(),
            linestyle=":",
            linewidth=1.5,
            label=fit_label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_h$")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        borderaxespad=0.0,
        labelspacing=0.25,
        handletextpad=0.4,
        frameon=False,
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def _extract_test_loss(run: wandb.apis.public.Run) -> float:
    history = run.history()
    if history is None or history.empty:
        raise RuntimeError(f"No history available for run '{run.name}'")

    test_loss_col = "test/loss"
    if test_loss_col in history.columns:
        test_series = history[test_loss_col].dropna()
        if len(test_series) > 0:
            return float(test_series.iloc[-1])

    test_cols = [c for c in history.columns if c.startswith("test/ngram_")]
    if not test_cols:
        raise RuntimeError(f"No test loss metrics found for run '{run.name}'")
    valid = history[test_cols].dropna(how="all")
    if valid.empty:
        raise RuntimeError(f"No finite test/ngram metrics found for run '{run.name}'")
    return float(valid.iloc[-1].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--max-hidden-dim",
        type=int,
        default=2**11,
        help="Maximum hidden_dim to include",
    )
    parser.add_argument(
        "--include-external",
        type=str,
        choices=["cagnetta", "kaplan", "shengqi", "shengi"],
        default=None,
        help=(
            "Use external L_n curve as d_h->infinity reference for n*_L "
            "instead of largest hidden_dim run"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    args = parser.parse_args()

    api = wandb.Api()
    group_runs = _resolve_group_runs(api, args.group)

    max_hidden_dim = None
    proxy_label: str
    if args.include_external is not None:
        inf_series = _load_external_l_reference(args.include_external)
        proxy_label = f"external:{args.include_external}"
    else:
        runs_by_hidden_dim = {
            int((run.config or {}).get("hidden_dim", -1)): run for run in group_runs
        }
        inf_series = None
        for candidate_hidden_dim in sorted(runs_by_hidden_dim.keys(), reverse=True):
            if candidate_hidden_dim < 0:
                continue
            candidate_series = _extract_combined_ngram_losses(
                runs_by_hidden_dim[candidate_hidden_dim]
            )
            if candidate_series:
                max_hidden_dim = candidate_hidden_dim
                inf_series = candidate_series
                break
        if max_hidden_dim is None or not inf_series:
            raise RuntimeError(
                f"No combined n-gram losses available in group '{args.group}' for d_h->infinity proxy"
            )
        proxy_label = f"hidden_dim={max_hidden_dim}"

    runs = [
        run
        for run in group_runs
        if int((run.config or {}).get("hidden_dim", -1)) <= args.max_hidden_dim
    ]
    if not runs:
        raise RuntimeError(
            f"No finished runs found for group='{args.group}' "
            f"with hidden_dim <= {args.max_hidden_dim}"
        )
    runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, args.group)

    combined_by_hidden_dim: dict[int, dict[int, float]] = {}
    test_loss_by_hidden_dim: dict[int, float] = {}
    for run in runs:
        hidden_dim = int((run.config or {})["hidden_dim"])
        combined_series = _extract_combined_ngram_losses(run)
        if not combined_series:
            raise RuntimeError(
                "No combined n-gram losses available for "
                f"run '{run.name}' (hidden_dim={hidden_dim})"
            )
        combined_by_hidden_dim[hidden_dim] = combined_series
        test_loss_by_hidden_dim[hidden_dim] = _extract_test_loss(run)

    inf_ns = np.array(sorted(inf_series.keys()), dtype=float)
    if inf_ns.size == 0:
        raise RuntimeError(
            f"No n values in d_h->infinity proxy series ({proxy_label})"
        )
    inf_values = np.array([float(inf_series[int(n)]) for n in inf_ns], dtype=float)

    rows: list[dict[str, float]] = []
    for run in tqdm(runs, desc="Runs", unit="run"):
        hidden_dim = int((run.config or {})["hidden_dim"])
        l_inf = _mean_last_n_values(combined_by_hidden_dim[hidden_dim])
        n_star = _interpolated_n_for_target_value(inf_series, l_inf)
        l_nstar_inf = float(np.interp(float(n_star), inf_ns, inf_values))
        l_combined_dh = _mean_combined_ngram_loss(combined_by_hidden_dim[hidden_dim])
        l_test_dh = float(test_loss_by_hidden_dim[hidden_dim])
        rows.append(
            {
                "hidden_dim": float(hidden_dim),
                "n_star": float(n_star),
                "l_inf": float(l_inf),
                "l_nstar_inf": l_nstar_inf,
                "l_combined_dh": float(l_combined_dh),
                "l_test_dh": l_test_dh,
            }
        )

    rows.sort(key=lambda row: row["hidden_dim"])
    print(f"Using {proxy_label} as d_h->infinity proxy")
    for row in rows:
        print(
            f"hidden_dim={row['hidden_dim']:.0f}, "
            f"n*_L={row['n_star']:.3f}, "
            f"L_inf(d_h)={row['l_inf']:.6g}, "
            f"L_n*_L(inf)={row['l_nstar_inf']:.6g}, "
            f"L_combined(d_h)={row['l_combined_dh']:.6g}, "
            f"L_test(d_h)={row['l_test_dh']:.6g}"
        )

    out_path = (
        args.output
        if args.output is not None
        else f"results/l_at_nstar_{args.group}.png"
    )
    _plot_curves(
        rows,
        out_path,
        title=(f"{args.group} (n*_L from L_n, d_h->infinity proxy={proxy_label})"),
    )


if __name__ == "__main__":
    main()

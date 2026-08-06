"""Plot L_infinity vs hidden_dim."""

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
import pandas as pd
from scipy.optimize import curve_fit

import wandb
from analysis.plot_group_labels import distinct_group_labels
from analysis.ngram_split_utils import ngram_prefixes_for_split

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


def power_law_no_offset(n, c, power):
    return c * np.power(n, power)


def _load_external_ln_data(source: str) -> tuple[np.ndarray, np.ndarray]:
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


def _external_h_inf(source: str, tail_points: int) -> float:
    _, losses = _load_external_ln_data(source)
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        raise RuntimeError(f"No losses in external source '{source}'")
    tail_n = min(int(tail_points), int(losses.size))
    if tail_n < 1:
        raise RuntimeError("tail_points must be >= 1")
    return float(np.mean(losses[-tail_n:]))


def fetch_runs(project: str, group: str | None = None) -> list[wandb.apis.public.Run]:
    api = wandb.Api(timeout=60)
    filters = {"group": group, "state": "finished"} if group else {"state": "finished"}
    runs = api.runs(project, filters=filters)
    return list(runs)


def _extract_ngram_index(metric_key: str) -> int | None:
    name = metric_key.split("/")[-1]
    if name.startswith("ngram_"):
        return int(name.split("_")[-1])
    if name.startswith("n_gram_"):
        return int(name.split("_")[-1])
    return None


def extract_l_ngrams(
    runs: list[wandb.apis.public.Run],
    *,
    split: str = "combined",
) -> pd.DataFrame:
    rows = []
    prefixes = ngram_prefixes_for_split(split)
    for run in runs:
        hidden_dim = run.config.get("hidden_dim")
        if hidden_dim is None:
            continue
        hidden_dim = int(hidden_dim)

        summary = run.summary or {}
        matched_keys = [
            key for key in summary.keys() if any(key.startswith(p) for p in prefixes)
        ]
        if matched_keys:
            for key in matched_keys:
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
        matched_keys = [
            key for key in history_cols if any(key.startswith(p) for p in prefixes)
        ]
        if not matched_keys:
            continue
        history = run.history(keys=matched_keys, samples=10000)
        if history.empty:
            continue
        valid = history[matched_keys].dropna(how="all")
        if valid.empty:
            continue
        last_row = valid.iloc[-1]
        for key in matched_keys:
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


L_INF_TAIL_POINTS_DEFAULT = 100


def fit_l_infinity(
    df: pd.DataFrame, tail_points: int = L_INF_TAIL_POINTS_DEFAULT
) -> pd.DataFrame:
    """L_infinity as the mean of L_n over the largest `tail_points` values of n.

    Previously this extrapolated the L_inf parameter of a
    L_n = L_inf + c*n^alpha fit over n <= 50. That reads an asymptote off the
    steepest part of the curve: L_n is still falling well past n = 1000, so the
    fitted intercept is an extrapolation far outside the fitted window and is
    sensitive to the window choice. Averaging the measured tail is a direct
    estimate of the same quantity, and matches the saturation convention used
    in analysis/plot_mi_saturation.py (_tail_averaged_mi_value).
    """
    rows = []
    for hidden_dim, group in df.groupby("hidden_dim"):
        group = group.sort_values("n")
        losses = group["loss"].to_numpy(dtype=float)
        finite = losses[np.isfinite(losses)]
        if finite.size == 0:
            continue
        tail = finite[-min(int(tail_points), finite.size) :]
        rows.append(
            {
                "hidden_dim": int(hidden_dim),
                "l_inf": float(np.mean(tail)),
                "n_tail": int(tail.size),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["hidden_dim", "l_inf", "n_tail"])
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
    l_inf_by_group: dict[str, pd.DataFrame],
    labels_by_group: dict[str, str],
    out_path: str,
    title: str | None,
    external_h_inf: float | None = None,
    raw: bool = False,
) -> None:
    non_empty = {k: v for k, v in l_inf_by_group.items() if not v.empty}
    if not non_empty:
        raise RuntimeError("No data found.")
    plt.figure(figsize=(8, 7.5))
    show_group_name = len(non_empty) > 1

    def _plot_with_fit(
        x: np.ndarray,
        y: np.ndarray,
        *,
        series_label: str,
        marker: str,
        label_base: str,
        asymptote_label: str,
        color: str | None = None,
        fixed_asymptote: float | None = None,
    ) -> float | None:
        if raw:
            label = f"{label_base}{' [' + series_label + ']' if show_group_name else ''}"
            kwargs = {
                "marker": marker,
                "markeredgecolor": "black",
                "alpha": 0.8,
                "label": label,
            }
            if color is not None:
                kwargs["color"] = color
            plt.plot(x, y, **kwargs)
            return None

        fit_mask = (x > 0) & (y > 0)
        nu = None
        power_fit = None
        asymptote = None
        coef = None
        x_fit = None
        y_fit = None
        if np.count_nonzero(fit_mask) >= 2:
            x_fit = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 200)

            if fixed_asymptote is None:
                coef0 = float(max(np.max(y[fit_mask]), 1e-8))
                p0 = [coef0, -0.5]
                try:
                    popt, _ = curve_fit(
                        power_law_no_offset,
                        x[fit_mask],
                        y[fit_mask],
                        p0=p0,
                        maxfev=10_000,
                        bounds=([0, -np.inf], [np.inf, 0]),
                    )
                    coef = float(popt[0])
                    power_fit = float(popt[1])
                    nu = -float(popt[1])
                    y_fit = power_law_no_offset(x_fit, *popt)
                except RuntimeError:
                    y_fit = None
            else:
                asymptote = float(fixed_asymptote)

                def _model_fixed(x_in, coef_in, power):
                    return asymptote + coef_in * np.power(x_in, power)

                coef0 = float(max(np.max(y[fit_mask]) - asymptote, 1e-8))
                p0 = [coef0, -0.5]
                try:
                    popt, _ = curve_fit(
                        _model_fixed,
                        x[fit_mask],
                        y[fit_mask],
                        p0=p0,
                        maxfev=10_000,
                        bounds=([-np.inf, -np.inf], [np.inf, 0]),
                    )
                    coef = float(popt[0])
                    power_fit = float(popt[1])
                    nu = -float(popt[1])
                    y_fit = _model_fixed(x_fit, *popt)
                except RuntimeError:
                    y_fit = None

        fit_terms = []
        if nu is not None:
            fit_terms.append(rf"$\nu$={nu:.3f}")
        legend_suffix = f" [{series_label}]" if show_group_name else ""
        if fixed_asymptote is not None and coef is not None and power_fit is not None:
            label = rf"$L_\infty(d_h)={coef:.3g}\times d_h^{{{power_fit:.3f}}}+H_\infty${legend_suffix}"
        elif fixed_asymptote is None and coef is not None and power_fit is not None:
            label = rf"$L_\infty(d_h)={coef:.3g}\times d_h^{{{power_fit:.3f}}}${legend_suffix}"
        else:
            base = f"{label_base}{legend_suffix}"
            label = f"{base} ({', '.join(fit_terms)})" if fit_terms else base
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

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    nus: list[float] = []
    for idx, (group_name, l_inf_df) in enumerate(non_empty.items()):
        group_label = labels_by_group.get(group_name, group_name) if show_group_name else ""
        nu_group = _plot_with_fit(
            l_inf_df["hidden_dim"].to_numpy(dtype=float),
            l_inf_df["l_inf"].to_numpy(dtype=float),
            series_label=group_label,
            marker=markers[idx % len(markers)],
            label_base=rf"$L_{{\infty}}(d_h)$",
            asymptote_label=r"$L_{\infty}(\infty)$",
            color=colors[idx % len(colors)],
            fixed_asymptote=external_h_inf,
        )
        if nu_group is not None:
            nus.append(float(nu_group))

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel(r"$d_h$")
    plt.ylabel("")
    if title:
        plt.title(title)
    if nus:
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=1,
            borderaxespad=0.0,
            labelspacing=0.25,
            handletextpad=0.4,
            frameon=False,
        )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    plt.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, nargs="+", default=None)
    parser.add_argument(
        "--split",
        type=str,
        choices=["combined", "validation", "train", "test", "all"],
        default="combined",
        help="Which n-gram split to plot; validation uses val/ngram_*",
    )
    parser.add_argument("--output", type=str, default="results/L_infinity.png")
    parser.add_argument(
        "--all-token-checkpoints",
        action="store_true",
        help=(
            "One series per token checkpoint, coloured by training tokens, "
            "read from the MI cache. The logged ngram metrics used by the "
            "default path have no per-checkpoint dimension."
        ),
    )
    parser.add_argument(
        "--hidden-dim", type=int, nargs="+", default=None,
        help="Optional hidden_dim filter for --all-token-checkpoints",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="checkpoints/bipartite_mi_cache",
        help="MI cache directory to read scored log-probs from",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size when scoring a checkpoint that is not cached yet",
    )
    parser.add_argument(
        "--force-resample", action="store_true",
        help="Rescore checkpoints even when cached log-probs exist",
    )

    parser.add_argument("--max-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--l-inf-tail-points",
        type=int,
        default=L_INF_TAIL_POINTS_DEFAULT,
        help=(
            "L_infinity is the mean of L_n over this many largest n "
            "(default %(default)s). Replaces the previous power-law "
            "extrapolation of the L_inf parameter from n <= 50."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw points only, without fitting power laws",
    )
    parser.add_argument(
        "--include-external",
        type=str,
        choices=["cagnetta", "kaplan", "shengqi", "shengi"],
        default=None,
        help="Use external ln source to define fixed H_infinity",
    )
    parser.add_argument(
        "--external-tail-points",
        type=int,
        default=20,
        help="Number of tail points from external curve to average for H_infinity",
    )
    args = parser.parse_args()

    if args.all_token_checkpoints:
        # Separate path: the logged ngram metrics the default path reads have
        # no per-checkpoint dimension, so L_n comes from the MI cache instead.
        from analysis.ngram_cache import (
            ngram_losses_by_token_checkpoint,
            plot_vs_hidden_dim_by_tokens,
        )

        if not args.group or len(args.group) != 1:
            raise RuntimeError(
                "--all-token-checkpoints requires exactly one --group"
            )
        ln = ngram_losses_by_token_checkpoint(
            args.group[0],
            hidden_dims=args.hidden_dim,
            cache_dir=args.cache_dir,
            data_split=(args.split if args.split in
                        {"validation", "test", "train"} else "validation"),
            batch_size=args.batch_size,
            force_resample=args.force_resample,
        )
        points = (
            ln.groupby(["hidden_dim", "tokens"], as_index=False)
            .apply(lambda g: pd.Series({
                "l_inf": float(g.sort_values("n")["loss"]
                               .to_numpy()[-args.l_inf_tail_points:].mean())
            }), include_groups=False)
            .reset_index(drop=True)
        )
        plot_vs_hidden_dim_by_tokens(
            points, "l_inf", r"$L_\infty$", args.output, fit=not args.raw
        )
        return


    groups = args.group if args.group else [None]
    group_names = [g for g in groups if g is not None]
    labels_by_group = distinct_group_labels(group_names)
    l_inf_by_group: dict[str, pd.DataFrame] = {}
    for group_name in groups:
        runs = fetch_runs("tarunadvaith-/scaling", group=group_name)
        l_n_df = extract_l_ngrams(runs, split=args.split)
        if args.max_hidden_dim is not None:
            l_n_df = l_n_df[l_n_df["hidden_dim"] <= args.max_hidden_dim]
        key = group_name if group_name is not None else "all"
        if args.raw:
            l_inf_by_group[key] = (
                l_n_df.sort_values(["hidden_dim", "n"])
                .groupby("hidden_dim", as_index=False)
                .tail(1)[["hidden_dim", "loss"]]
                .rename(columns={"loss": "l_inf"})
                .sort_values("hidden_dim")
            )
        else:
            l_inf_by_group[key] = fit_l_infinity(
                l_n_df, tail_points=args.l_inf_tail_points
            )
        if key not in labels_by_group:
            labels_by_group[key] = key

    title = ", ".join(groups) if args.group else None
    if args.split != "combined":
        title = f"{title} [{args.split}]" if title else args.split
        if args.output == "results/L_infinity.png":
            args.output = args.output.replace(".png", f"_{args.split}.png")
    external_h_inf = None
    if args.include_external is not None:
        external_h_inf = _external_h_inf(
            args.include_external, args.external_tail_points
        )
        print(
            f"Using H_infinity from external source '{args.include_external}': "
            f"H_infinity={external_h_inf:.6g} "
            f"(tail_points={args.external_tail_points})"
        )
    plot_l_infinity(
        l_inf_by_group,
        labels_by_group,
        args.output,
        title,
        external_h_inf=external_h_inf,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()

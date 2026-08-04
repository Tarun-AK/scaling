"""Plot L vs hidden_dim."""

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

from analysis.ngram_split_utils import ngram_prefixes_for_split
from analysis.plot_group_labels import distinct_group_labels

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


def power_law(n, l_value, c, power):
    return l_value + c * np.power(n, power)


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


def _external_h(source: str) -> float:
    _, losses = _load_external_ln_data(source)
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        raise RuntimeError(f"No losses in external source '{source}'")
    return float(np.mean(losses))


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
        combined_keys = [
            key for key in summary.keys() if any(key.startswith(p) for p in prefixes)
        ]
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
        combined_keys = [
            key for key in history_cols if any(key.startswith(p) for p in prefixes)
        ]
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


def compute_l(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hidden_dim", "l"])
    return (
        df.groupby("hidden_dim", as_index=False)[["loss"]]
        .mean()
        .rename(columns={"loss": "l"})
        .sort_values("hidden_dim")
    )


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


def plot_l(
    l_by_group: dict[str, pd.DataFrame],
    out_path: str,
    title: str | None,
    external_h: float | None = None,
    labels_by_group: dict[str, str] | None = None,
    raw: bool = False,
) -> None:
    non_empty = {k: v for k, v in l_by_group.items() if not v.empty}
    if not non_empty:
        raise RuntimeError("No data found.")
    plt.figure(figsize=(8, 7.5))

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
        include_group_name_in_legend: bool = False,
    ) -> float | None:
        if raw:
            legend_suffix = f" [{series_label}]" if include_group_name_in_legend else ""
            label = f"{label_base}{legend_suffix}"
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
        legend_suffix = f" [{series_label}]" if include_group_name_in_legend else ""
        if fixed_asymptote is not None and coef is not None and power_fit is not None:
            label = rf"$L(d_h)={coef:.3g}\times d_h^{{{power_fit:.3f}}}+H${legend_suffix}"
        elif fixed_asymptote is None and coef is not None and power_fit is not None:
            label = rf"$L(d_h)={coef:.3g}\times d_h^{{{power_fit:.3f}}}${legend_suffix}"
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
    show_group_name = len(non_empty) > 1
    labels_by_group = labels_by_group or {}
    for idx, (group_name, l_df) in enumerate(non_empty.items()):
        group_label = labels_by_group.get(group_name, group_name) if show_group_name else ""
        nu_group = _plot_with_fit(
            l_df["hidden_dim"].to_numpy(dtype=float),
            l_df["l"].to_numpy(dtype=float),
            series_label=group_label,
            marker=markers[idx % len(markers)],
            label_base=rf"$L(d_h)$",
            asymptote_label=r"$L(\infty)$",
            color=colors[idx % len(colors)],
            fixed_asymptote=external_h,
            include_group_name_in_legend=show_group_name,
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
    parser.add_argument("--output", type=str, default="results/L.png")
    parser.add_argument("--max-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--include-external",
        type=str,
        choices=["cagnetta", "kaplan", "shengqi", "shengi"],
        default=None,
        help="Use external ln source to define fixed H",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw points only, without fitting power laws",
    )
    args = parser.parse_args()

    groups = args.group if args.group else [None]
    l_by_group: dict[str, pd.DataFrame] = {}
    for group_name in groups:
        runs = fetch_runs("tarunadvaith-/scaling", group=group_name)
        l_n_df = extract_l_ngrams(runs, split=args.split)
        if args.max_hidden_dim is not None:
            l_n_df = l_n_df[l_n_df["hidden_dim"] <= args.max_hidden_dim]
        key = group_name if group_name is not None else "all"
        l_by_group[key] = compute_l(l_n_df)
        if l_by_group[key].empty:
            group_text = group_name if group_name is not None else "all runs"
            raise RuntimeError(
                f"No {args.split} n-gram losses found for group '{group_text}'."
            )

    title = ", ".join(groups) if args.group else None
    if args.split != "combined":
        title = f"{title} [{args.split}]" if title else args.split
        if args.output == "results/L.png":
            args.output = args.output.replace(".png", f"_{args.split}.png")
    labels_by_group = distinct_group_labels([g for g in groups if g is not None])
    external_h = None
    if args.include_external is not None:
        external_h = _external_h(args.include_external)
        print(
            f"Using H from external source '{args.include_external}': "
            f"H={external_h:.6g}"
        )
    plot_l(
        l_by_group,
        args.output,
        title,
        external_h=external_h,
        labels_by_group=labels_by_group,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()

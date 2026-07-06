"""Plot critical n* curves vs hidden_dim using sampled MI, L_n, and H_n."""

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
from scipy.optimize import curve_fit
from tqdm import tqdm

import wandb
from analysis.plot_bipartite_mi import (
    DEFAULT_MAX_N,
    DEFAULT_MIN_N,
    DEFAULT_N_VALUES,
    DEFAULT_NUM_N_VALUES,
    _compute_lstm_sampled_mi_for_run,
    _compute_lstm_v_club_for_run,
    _load_external_mi_data,
)

WANDB_PROJECT = "tarunadvaith-/scaling"
FIT_NMAX = 10
TAIL_MEAN_COUNT = 32
MI_TARGET_N = 1600
CURVE_TARGETS = {
    "mi": {"mi"},
    "l": {"l"},
    "h": {"h"},
    "all": {"mi", "l", "h"},
    "lh": {"l", "h"},
    "lmi": {"l", "mi"},
}
plt.style.use("~/plotStyle.mplstyle")


def _pure_power_law(x: np.ndarray, coef: float, power: float) -> np.ndarray:
    return coef * np.power(x, power)


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Fit y = coef * x^power in log-log space.

    Returns:
        Tuple of (coef, power, power_stderr) or None if fit is not possible.
    """

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None

    logx = np.log(x)
    logy = np.log(y)
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


def _resolve_group_runs(api: wandb.Api, group: str) -> list[wandb.apis.public.Run]:
    runs = list(
        api.runs(
            WANDB_PROJECT,
            filters={
                "group": group,
                "state": "finished",
            },
        )
    )
    if not runs:
        raise RuntimeError(f"No finished runs found for group='{group}'")

    by_hidden_dim: dict[int, wandb.apis.public.Run] = {}
    for run in tqdm(runs, desc="Runs", unit="run"):
        cfg = run.config or {}
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            continue
        hidden_dim = int(hidden_dim)
        if hidden_dim in by_hidden_dim:
            raise RuntimeError(
                "Multiple finished runs found for "
                f"group='{group}' hidden_dim={hidden_dim}"
            )
        by_hidden_dim[hidden_dim] = run

    if not by_hidden_dim:
        raise RuntimeError(
            f"No finished runs with config.hidden_dim found for group='{group}'"
        )

    return [by_hidden_dim[k] for k in sorted(by_hidden_dim)]


def _fit_asymptote(series: dict[int, float], fit_nmax: int = FIT_NMAX) -> float:
    if not series:
        raise RuntimeError("Cannot fit asymptote from empty series")

    ns = np.array(sorted(series.keys()), dtype=float)
    values = np.array([series[int(n)] for n in ns], dtype=float)

    fit_mask = ns <= float(fit_nmax)
    ns_fit = ns[fit_mask]
    values_fit = values[fit_mask]
    if len(ns_fit) < 2:
        ns_fit = ns
        values_fit = values
    if len(ns_fit) < 2:
        return float(values_fit[-1])

    p0 = [values_fit[-1], max(values_fit[0] - values_fit[-1], 1e-8), -0.5]
    try:
        popt, _ = curve_fit(
            lambda n_in, l_inf, c, power: l_inf + c * np.power(n_in, power),
            ns_fit,
            values_fit,
            p0=p0,
            maxfev=10_000,
            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0]),
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return float(np.min(values_fit))


def _solve_power_law_n(
    target: float, coef: float, power: float, offset: float
) -> float:
    if coef == 0.0:
        raise RuntimeError("Power-law coefficient cannot be zero")
    if power == 0.0:
        raise RuntimeError("Power-law exponent cannot be zero")

    ratio = (target - offset) / coef
    if ratio <= 0.0:
        raise RuntimeError(
            "Invalid target/reference combination for power-law inversion: "
            f"target={target}, coef={coef}, power={power}, offset={offset}"
        )

    n_star = float(ratio ** (1.0 / power))
    if not np.isfinite(n_star) or n_star <= 0.0:
        raise RuntimeError(
            "Invalid n* from power-law inversion: "
            f"target={target}, coef={coef}, power={power}, offset={offset}"
        )
    return n_star


def _mean_last_n_values(
    series: dict[int, float], tail_count: int = TAIL_MEAN_COUNT
) -> float:
    if not series:
        raise RuntimeError("Cannot compute tail mean from empty series")
    tail_ns = sorted(series.keys())[-int(tail_count) :]
    values = np.array([float(series[n]) for n in tail_ns], dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise RuntimeError("Cannot compute tail mean from non-finite series values")
    return float(np.mean(finite_values))


def _value_at_n(series: dict[int, float], n: int) -> float:
    if n not in series:
        raise RuntimeError(f"Missing value at N={n}")
    value = float(series[n])
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite value at N={n}")
    return value


def _interpolated_n_for_target_value(series: dict[int, float], target: float) -> float:
    if not series:
        raise RuntimeError("Cannot match target against empty n-gram series")

    ns = np.array(sorted(series.keys()), dtype=float)
    values = np.array([float(series[int(n)]) for n in ns], dtype=float)
    finite_mask = np.isfinite(values)
    ns = ns[finite_mask]
    values = values[finite_mask]
    if ns.size == 0:
        raise RuntimeError("No finite n-gram values available for target matching")
    if ns.size == 1:
        return float(ns[0])

    # Enforce monotone non-increasing L_n curve for stable inverse interpolation.
    values_mono = np.minimum.accumulate(values)

    # Invert monotone mapping L_n -> n by interpolation.
    # values_mono is non-increasing in n, so reverse to get increasing xp for np.interp.
    n_interp = float(
        np.interp(
            float(target),
            values_mono[::-1],
            ns[::-1],
            left=float(ns[-1]),
            right=float(ns[0]),
        )
    )
    return n_interp


def _interpolated_n_for_target_value_non_decreasing(
    series: dict[int, float], target: float
) -> float:
    if not series:
        raise RuntimeError("Cannot match target against empty series")

    ns = np.array(sorted(series.keys()), dtype=float)
    values = np.array([float(series[int(n)]) for n in ns], dtype=float)
    finite_mask = np.isfinite(values)
    ns = ns[finite_mask]
    values = values[finite_mask]
    if ns.size == 0:
        raise RuntimeError("No finite series values available for target matching")
    if ns.size == 1:
        return float(ns[0])

    values_mono = np.maximum.accumulate(values)
    n_interp = float(
        np.interp(
            float(target),
            values_mono,
            ns,
            left=float(ns[0]),
            right=float(ns[-1]),
        )
    )
    return n_interp


def _extract_ngram_index(metric_key: str) -> int | None:
    name = metric_key.split("/")[-1]
    if (
        name.startswith("ngram_")
        or name.startswith("n_gram_")
        or name.startswith("entropy_")
    ):
        try:
            return int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _combined_ngram_keys(keys: list[str]) -> list[str]:
    return [
        key
        for key in keys
        if key.startswith("combined/ngram_") or key.startswith("combined/n_gram_")
    ]


def _conditional_entropy_ngram_keys(keys: list[str]) -> list[str]:
    return [
        key
        for key in keys
        if key.startswith("conditional_entropy/entropy_")
        or key.startswith("conditional_entropy/ngram_")
        or key.startswith("conditional_entropy/n_gram_")
    ]


def _extract_combined_ngram_losses(run: wandb.apis.public.Run) -> dict[int, float]:
    out: dict[int, float] = {}

    summary = run.summary or {}
    combined_keys = _combined_ngram_keys(list(summary.keys()))
    for key in combined_keys:
        n = _extract_ngram_index(key)
        value = summary.get(key)
        if n is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[n] = val
    if out:
        return out

    history_cols = list(run.history(samples=1).columns)
    combined_keys = _combined_ngram_keys(history_cols)
    if not combined_keys:
        return out
    history = run.history(keys=combined_keys, samples=10_000)
    if history.empty:
        return out

    valid = history[combined_keys].dropna(how="all")
    if valid.empty:
        return out
    last_row = valid.iloc[-1]
    for key in combined_keys:
        n = _extract_ngram_index(key)
        value = last_row.get(key)
        if n is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[n] = val
    return out


def _load_external_l_reference(source: str) -> dict[int, float]:
    source_map = {
        "cagnetta": "cagnetta_ln.csv",
        "kaplan": "kaplan_ln.csv",
        "shengqi": "shengqi_ln.csv",
        "shengi": "shengqi_ln.csv",
    }
    if source not in source_map:
        raise RuntimeError(f"Unsupported external L reference source: {source}")

    primary = source_map[source]
    candidates = [primary]
    if source == "shengqi":
        candidates.append("shengi_ln.csv")
    if source == "shengi":
        candidates.append("shengqi_ln.csv")

    data_path = None
    for filename in candidates:
        candidate_path = os.path.join(os.path.dirname(__file__), "..", "externalData", filename)
        if os.path.exists(candidate_path):
            data_path = candidate_path
            break
    if data_path is None:
        raise FileNotFoundError(
            "Missing external L_n file. Looked for: " + ", ".join(candidates)
        )

    data = np.loadtxt(data_path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"Unexpected external data shape for '{source}': {data.shape}"
        )

    series: dict[int, float] = {}
    for row in data:
        n = int(round(float(row[0])))
        value = float(row[1])
        if n > 0 and np.isfinite(value):
            series[n] = value
    if not series:
        raise RuntimeError(f"No valid points found in external data '{source}'")
    return series


def _load_external_mi_reference(source: str) -> dict[int, float]:
    ns_external, mi_external = _load_external_mi_data(source)
    series: dict[int, float] = {}
    for n_raw, value_raw in zip(ns_external, mi_external, strict=False):
        n = int(round(float(n_raw)))
        value = float(value_raw)
        if n > 0 and np.isfinite(value):
            series[n] = value
    if not series:
        raise RuntimeError(f"No valid points found in external MI data '{source}'")
    return series


def _extract_conditional_entropy_ngram_losses(
    run: wandb.apis.public.Run,
) -> dict[int, float]:
    out: dict[int, float] = {}

    summary = run.summary or {}
    entropy_ngram_keys = _conditional_entropy_ngram_keys(list(summary.keys()))
    for key in entropy_ngram_keys:
        n = _extract_ngram_index(key)
        value = summary.get(key)
        if n is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[n] = val
    if out:
        return out

    history_cols = list(run.history(samples=1).columns)
    entropy_ngram_keys = _conditional_entropy_ngram_keys(history_cols)
    if not entropy_ngram_keys:
        return out
    history = run.history(keys=entropy_ngram_keys, samples=10_000)
    if history.empty:
        return out

    valid = history[entropy_ngram_keys].dropna(how="all")
    if valid.empty:
        return out
    last_row = valid.iloc[-1]
    for key in entropy_ngram_keys:
        n = _extract_ngram_index(key)
        value = last_row.get(key)
        if n is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[n] = val
    return out


def _compute_sampled_mi_series(
    *,
    run: wandb.apis.public.Run,
    api: wandb.Api,
    n_values: list[int],
    mi_estimator: str,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> dict[int, float]:
    hidden_dim = int((run.config or {})["hidden_dim"])
    if mi_estimator == "direct":
        return _compute_lstm_sampled_mi_for_run(
            run,
            api,
            hidden_dim,
            n_values,
            num_samples=num_samples,
            batch_size=batch_size,
            cache_dir=cache_dir,
            force_resample=force_resample,
        )
    if mi_estimator == "v-club":
        return _compute_lstm_v_club_for_run(
            run,
            api,
            hidden_dim,
            n_values,
            num_samples=num_samples,
            batch_size=batch_size,
            cache_dir=cache_dir,
            force_resample=force_resample,
        )
    raise RuntimeError(f"Unsupported --mi-estimator: {mi_estimator}")


def _compute_n_star_rows(
    *,
    runs: list[wandb.apis.public.Run],
    api: wandb.Api,
    n_values: list[int],
    curve: str,
    l_ref_coef: float,
    l_ref_power: float,
    l_ref_offset: float,
    l_ref_mode: str,
    h_ref_coef: float,
    h_ref_power: float,
    h_ref_offset: float,
    mi_estimator: str,
    mi_ref_mode: str,
    mi_ref_coef: float,
    mi_ref_power: float,
    mi_ref_offset: float,
    mi_inf_proxy_hidden_dim: int | None,
    mi_inf_proxy_series: dict[int, float] | None,
    mi_inf_proxy_source: str | None,
    l_inf_proxy_hidden_dim: int | None,
    l_inf_proxy_series: dict[int, float] | None,
    l_inf_proxy_source: str | None,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> tuple[list[dict[str, float]], str, str, str]:
    active_targets = CURVE_TARGETS[curve]
    mi_by_hidden_dim: dict[int, dict[int, float]] = {}
    l_by_hidden_dim: dict[int, dict[int, float]] = {}
    h_by_hidden_dim: dict[int, dict[int, float]] = {}

    for run in tqdm(runs, desc="Runs", unit="run"):
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])

        if "mi" in active_targets:
            mi_series = _compute_sampled_mi_series(
                run=run,
                api=api,
                n_values=n_values,
                mi_estimator=mi_estimator,
                num_samples=num_samples,
                batch_size=batch_size,
                cache_dir=cache_dir,
                force_resample=force_resample,
            )
            if not mi_series:
                raise RuntimeError(
                    "No sampled bipartite MI series available for "
                    f"run '{run.name}' (hidden_dim={hidden_dim})"
                )
            mi_by_hidden_dim[hidden_dim] = mi_series

        if "l" in active_targets:
            l_series = _extract_combined_ngram_losses(run)
            if not l_series:
                raise RuntimeError(
                    "No combined n-gram losses available for "
                    f"run '{run.name}' (hidden_dim={hidden_dim})"
                )
            l_by_hidden_dim[hidden_dim] = l_series

        if "h" in active_targets:
            h_series = _extract_conditional_entropy_ngram_losses(run)
            if not h_series:
                raise RuntimeError(
                    "No conditional-entropy metrics available for "
                    f"run '{run.name}' (hidden_dim={hidden_dim})"
                )
            h_by_hidden_dim[hidden_dim] = h_series

    if "mi" in active_targets:
        print(f"{mi_estimator} MI values by hidden_dim:")
        for hidden_dim in sorted(mi_by_hidden_dim):
            mi_series = mi_by_hidden_dim[hidden_dim]
            if not mi_series:
                continue
            mi_target_value = _value_at_n(mi_series, MI_TARGET_N)
            print(
                f"  hidden_dim={hidden_dim}: "
                f"MI(N={MI_TARGET_N})={mi_target_value:.6g}"
            )

    hidden_dim_sets: list[set[int]] = []
    if "mi" in active_targets:
        hidden_dim_sets.append(set(mi_by_hidden_dim.keys()))
    if "l" in active_targets:
        hidden_dim_sets.append(set(l_by_hidden_dim.keys()))
    if "h" in active_targets:
        hidden_dim_sets.append(set(h_by_hidden_dim.keys()))
    selected_hidden_dims = sorted(set.intersection(*hidden_dim_sets))
    if not selected_hidden_dims:
        if curve == "both":
            raise RuntimeError("No hidden_dim has both sampled MI and L_n series")
        if curve == "all":
            raise RuntimeError("No hidden_dim has sampled MI, L_n, and H_n series")
        if curve == "mi":
            raise RuntimeError("No hidden_dim has sampled MI series")
        if curve == "l":
            raise RuntimeError("No hidden_dim has L_n series")
        raise RuntimeError("No hidden_dim has H_n series")

    l_inf_curve_hidden_dim = l_inf_proxy_hidden_dim
    l_inf_curve = l_inf_proxy_series
    if "l" in active_targets and l_ref_mode == "proxy-curve" and not l_inf_curve:
        raise RuntimeError(
            "Missing combined n-gram series for d_h->infinity proxy hidden_dim"
        )

    mi_inf_curve_hidden_dim = mi_inf_proxy_hidden_dim
    mi_inf_curve = mi_inf_proxy_series
    if "mi" in active_targets and mi_ref_mode == "proxy-curve" and not mi_inf_curve:
        raise RuntimeError("Missing MI series for d_h->infinity proxy hidden_dim")

    mi_reference_label = (
        (
            f"analytic_power_law(coef={mi_ref_coef}, power={mi_ref_power}, offset={mi_ref_offset})"
            if mi_ref_mode == "analytic-power-law"
            else (
                f"external_curve(source={mi_inf_proxy_source})"
                if mi_inf_proxy_source is not None
                else f"empirical_inf_curve(hidden_dim={mi_inf_curve_hidden_dim})"
            )
        )
        if "mi" in active_targets
        else "disabled"
    )
    l_reference_label = (
        (
            f"analytic_power_law(coef={l_ref_coef}, power={l_ref_power}, offset={l_ref_offset})"
            if l_ref_mode == "analytic-power-law"
            else (
                f"external_curve(source={l_inf_proxy_source})"
                if l_inf_proxy_source is not None
                else f"empirical_inf_curve(hidden_dim={l_inf_curve_hidden_dim})"
            )
        )
        if "l" in active_targets
        else "disabled"
    )
    h_reference_label = (
        f"power_law(coef={h_ref_coef}, power={h_ref_power}, offset={h_ref_offset})"
        if "h" in active_targets
        else "disabled"
    )

    rows: list[dict[str, float]] = []
    for hidden_dim in tqdm(selected_hidden_dims, desc="Computing n*", unit="run"):
        row: dict[str, float] = {"hidden_dim": float(hidden_dim)}

        if "mi" in active_targets:
            mi_inf = _value_at_n(mi_by_hidden_dim[hidden_dim], MI_TARGET_N)
            if mi_ref_mode == "analytic-power-law":
                n_star_i = _solve_power_law_n(
                    target=mi_inf,
                    coef=mi_ref_coef,
                    power=mi_ref_power,
                    offset=mi_ref_offset,
                )
            else:
                if mi_inf_curve is None:
                    raise RuntimeError("Internal error: missing d_h->infinity MI curve")
                n_star_i = _interpolated_n_for_target_value_non_decreasing(
                    mi_inf_curve,
                    mi_inf,
                )
            row["mi_inf"] = mi_inf
            row["n_star_i"] = float(n_star_i)

        if "l" in active_targets:
            l_inf = _mean_last_n_values(l_by_hidden_dim[hidden_dim], TAIL_MEAN_COUNT)
            if l_ref_mode == "analytic-power-law":
                n_star_l = _solve_power_law_n(
                    target=l_inf,
                    coef=l_ref_coef,
                    power=l_ref_power,
                    offset=l_ref_offset,
                )
            else:
                if l_inf_curve is None:
                    raise RuntimeError("Internal error: missing d_h->infinity L_n curve")
                n_star_l = _interpolated_n_for_target_value(l_inf_curve, l_inf)
            row["l_inf"] = l_inf
            row["n_star_l"] = float(n_star_l)

        if "h" in active_targets:
            h_inf = _mean_last_n_values(h_by_hidden_dim[hidden_dim], TAIL_MEAN_COUNT)
            n_star_h = _solve_power_law_n(
                target=h_inf,
                coef=h_ref_coef,
                power=h_ref_power,
                offset=h_ref_offset,
            )
            row["h_inf"] = h_inf
            row["n_star_h"] = float(n_star_h)

        rows.append(row)

    return rows, mi_reference_label, l_reference_label, h_reference_label


def _plot_n_star(
    rows: list[dict[str, float]],
    out_path: str,
    title: str,
    curve: str,
    *,
    fit: bool,
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot")
    active_targets = CURVE_TARGETS[curve]

    hidden_dims = np.array([row["hidden_dim"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    if "mi" in active_targets:
        n_star_i = np.array([row["n_star_i"] for row in rows], dtype=float)
        (line,) = ax.plot(
            hidden_dims,
            n_star_i,
            marker="o",
            markeredgecolor="black",
            linestyle="-",
            label=r"$n^*_{MI}$",
        )
        if fit:
            fit_result = _fit(hidden_dims, n_star_i)
            if fit_result is not None:
                coef, power, power_stderr = fit_result
                x_fit = np.geomspace(hidden_dims.min(), hidden_dims.max(), 200)
                y_fit = _pure_power_law(x_fit, coef, power)
                if np.isfinite(power_stderr):
                    fit_label = (
                        rf"fit MI: $n^*={coef:.3g}\,d_h^{{{power:.3f}}}$, "
                        rf"$b={power:.3f}\pm{power_stderr:.3f}$"
                    )
                else:
                    fit_label = rf"fit MI: $n^*={coef:.3g}\,d_h^{{{power:.3f}}}$"
                ax.plot(
                    x_fit,
                    y_fit,
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=1.5,
                    label=fit_label,
                )
    if "l" in active_targets:
        n_star_l = np.array([row["n_star_l"] for row in rows], dtype=float)
        (line,) = ax.plot(
            hidden_dims,
            n_star_l,
            marker="s",
            markeredgecolor="black",
            linestyle="--",
            label=r"$n^*_{L_n}$",
        )
        if fit:
            fit_result = _fit(hidden_dims, n_star_l)
            if fit_result is not None:
                coef, power, power_stderr = fit_result
                x_fit = np.geomspace(hidden_dims.min(), hidden_dims.max(), 200)
                y_fit = _pure_power_law(x_fit, coef, power)
                if np.isfinite(power_stderr):
                    fit_label = (
                        rf"fit L: $n^*={coef:.3g}\,d_h^{{{power:.3f}}}$, "
                        rf"$b={power:.3f}\pm{power_stderr:.3f}$"
                    )
                else:
                    fit_label = rf"fit L: $n^*={coef:.3g}\,d_h^{{{power:.3f}}}$"
                ax.plot(
                    x_fit,
                    y_fit,
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=1.5,
                    label=fit_label,
                )
    if "h" in active_targets:
        n_star_h = np.array([row["n_star_h"] for row in rows], dtype=float)
        (line,) = ax.plot(
            hidden_dims,
            n_star_h,
            marker="d",
            markeredgecolor="black",
            linestyle="-.",
            label=r"$n^*_{H_n}$",
        )
        if fit:
            fit_result = _fit(hidden_dims, n_star_h)
            if fit_result is not None:
                coef, power, power_stderr = fit_result
                x_fit = np.geomspace(hidden_dims.min(), hidden_dims.max(), 200)
                y_fit = _pure_power_law(x_fit, coef, power)
                if np.isfinite(power_stderr):
                    fit_label = (
                        rf"fit H: $n^*={coef:.3g}\,d_h^{{{power:.3f}}}$, "
                        rf"$b={power:.3f}\pm{power_stderr:.3f}$"
                    )
                else:
                    fit_label = rf"fit H: $n^*={coef:.3g}\,d_h^{{{power:.3f}}}$"
                ax.plot(
                    x_fit,
                    y_fit,
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=1.5,
                    label=fit_label,
                )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$d_h$")
    ax.set_ylabel(r"$n^*$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument(
        "--curve",
        type=str,
        choices=["mi", "l", "h", "all", "lh", "lmi"],
        default="all",
        help="Which n* curve(s) to compute and plot",
    )
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
        "--l-ref-coef",
        type=float,
        default=3.6,
        help="Power-law coefficient for analytic L reference",
    )
    parser.add_argument(
        "--l-ref-power",
        type=float,
        default=-1.098,
        help="Power-law exponent for analytic L reference",
    )
    parser.add_argument(
        "--l-ref-offset",
        type=float,
        default=3.71,
        help="Power-law offset for analytic L reference",
    )
    parser.add_argument(
        "--l-ref-mode",
        type=str,
        choices=["proxy-curve", "analytic-power-law"],
        default="proxy-curve",
        help=(
            "How to invert L to n*: use d_h->infinity proxy curve "
            "(external or largest hidden_dim) or analytic power law"
        ),
    )
    parser.add_argument(
        "--include-external",
        type=str,
        choices=["cagnetta", "kaplan", "shengqi", "shengi"],
        default=None,
        help=(
            "Use external MI/L_n curves as d_h->infinity references for n*_I/n*_L "
            "instead of largest hidden_dim run"
        ),
    )
    parser.add_argument(
        "--h-ref-coef",
        type=float,
        default=3.86,
        help="Power-law coefficient for H reference",
    )
    parser.add_argument(
        "--h-ref-power",
        type=float,
        default=-0.950,
        help="Power-law exponent for H reference",
    )
    parser.add_argument(
        "--h-ref-offset",
        type=float,
        default=2.98,
        help="Power-law offset for H reference",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="Maximum N to include in sampled MI curve",
    )
    parser.add_argument(
        "--mi-estimator",
        type=str,
        choices=["direct", "v-club"],
        default="direct",
        help="MI estimator to use for MI-based n* extraction",
    )
    parser.add_argument(
        "--mi-ref-mode",
        type=str,
        choices=["proxy-curve", "analytic-power-law"],
        default="proxy-curve",
        help=(
            "How to invert MI to n*: use d_h->infinity proxy curve "
            "(external or largest hidden_dim) or analytic power law"
        ),
    )
    parser.add_argument(
        "--mi-ref-coef",
        type=float,
        default=2.41,
        help="Power-law coefficient for analytic MI reference",
    )
    parser.add_argument(
        "--mi-ref-power",
        type=float,
        default=0.366,
        help="Power-law exponent for analytic MI reference",
    )
    parser.add_argument(
        "--mi-ref-offset",
        type=float,
        default=0.0,
        help="Power-law offset for analytic MI reference",
    )
    parser.add_argument(
        "--num-n-values",
        type=int,
        default=DEFAULT_NUM_N_VALUES,
        help="Number of log-spaced N values for sampled MI curve",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of sampled sequences for sampled MI",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sampling/scoring sampled MI",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
        help="Directory for sampled MI cache",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Force regeneration of sampled caches instead of cache-only mode",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help=r"Fit and overlay pure power-law fits: $n^*(d_h)=a\,d_h^b$",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    args = parser.parse_args()

    if args.max_n < DEFAULT_MIN_N:
        raise RuntimeError(f"--max-n must be >= {DEFAULT_MIN_N}")
    if args.num_n_values < 1:
        raise RuntimeError("--num-n-values must be >= 1")

    n_values = [n for n in DEFAULT_N_VALUES if n <= int(args.max_n)]
    if not n_values:
        raise RuntimeError("No valid N values to evaluate")
    if args.curve in {"mi", "all", "lmi"} and int(args.max_n) < MI_TARGET_N:
        raise RuntimeError(
            f"--max-n must be >= {MI_TARGET_N} for MI n* (uses MI at N={MI_TARGET_N})"
        )

    api = wandb.Api()
    group_runs = _resolve_group_runs(api, args.group)

    mi_inf_proxy_hidden_dim: int | None = None
    mi_inf_proxy_series: dict[int, float] | None = None
    mi_inf_proxy_source: str | None = None
    if args.curve in {"mi", "all", "lmi"} and args.mi_ref_mode == "proxy-curve":
        if args.include_external is not None:
            mi_inf_proxy_series = _load_external_mi_reference(args.include_external)
            mi_inf_proxy_source = args.include_external
        else:
            runs_by_hidden_dim = {
                int((run.config or {}).get("hidden_dim", -1)): run
                for run in group_runs
            }
            for candidate_hidden_dim in sorted(runs_by_hidden_dim.keys(), reverse=True):
                if candidate_hidden_dim < 0:
                    continue
                candidate_series = _compute_sampled_mi_series(
                    run=runs_by_hidden_dim[candidate_hidden_dim],
                    api=api,
                    n_values=n_values,
                    mi_estimator=args.mi_estimator,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                )
                if candidate_series:
                    mi_inf_proxy_hidden_dim = candidate_hidden_dim
                    mi_inf_proxy_series = candidate_series
                    break
            if mi_inf_proxy_hidden_dim is None or not mi_inf_proxy_series:
                raise RuntimeError(
                    "No MI series available for any run in group "
                    f"'{args.group}' to use as d_h->infinity proxy"
                )

    l_inf_proxy_hidden_dim: int | None = None
    l_inf_proxy_series: dict[int, float] | None = None
    l_inf_proxy_source: str | None = None
    if args.curve in {"l", "lh", "all", "lmi"} and args.l_ref_mode == "proxy-curve":
        if args.include_external is not None:
            l_inf_proxy_series = _load_external_l_reference(args.include_external)
            l_inf_proxy_source = args.include_external
        else:
            runs_by_hidden_dim = {
                int((run.config or {}).get("hidden_dim", -1)): run
                for run in group_runs
            }
            for candidate_hidden_dim in sorted(runs_by_hidden_dim.keys(), reverse=True):
                if candidate_hidden_dim < 0:
                    continue
                candidate_series = _extract_combined_ngram_losses(
                    runs_by_hidden_dim[candidate_hidden_dim]
                )
                if candidate_series:
                    l_inf_proxy_hidden_dim = candidate_hidden_dim
                    l_inf_proxy_series = candidate_series
                    break
            if l_inf_proxy_hidden_dim is None or not l_inf_proxy_series:
                raise RuntimeError(
                    "No combined n-gram losses available for any run in group "
                    f"'{args.group}' to use as d_h->infinity proxy"
                )

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
    if args.hidden_dim is not None:
        hidden_dims = set(args.hidden_dim)
        runs = [
            run
            for run in runs
            if int((run.config or {}).get("hidden_dim", -1)) in hidden_dims
        ]
        if not runs:
            raise RuntimeError(
                f"No finished runs found for group='{args.group}' "
                f"hidden_dim in {sorted(hidden_dims)}"
            )

    rows, mi_ref_label, l_ref_label, h_ref_label = _compute_n_star_rows(
        runs=runs,
        api=api,
        n_values=n_values,
        curve=args.curve,
        l_ref_coef=args.l_ref_coef,
        l_ref_power=args.l_ref_power,
        l_ref_offset=args.l_ref_offset,
        l_ref_mode=args.l_ref_mode,
        h_ref_coef=args.h_ref_coef,
        h_ref_power=args.h_ref_power,
        h_ref_offset=args.h_ref_offset,
        mi_estimator=args.mi_estimator,
        mi_ref_mode=args.mi_ref_mode,
        mi_ref_coef=args.mi_ref_coef,
        mi_ref_power=args.mi_ref_power,
        mi_ref_offset=args.mi_ref_offset,
        mi_inf_proxy_hidden_dim=mi_inf_proxy_hidden_dim,
        mi_inf_proxy_series=mi_inf_proxy_series,
        mi_inf_proxy_source=mi_inf_proxy_source,
        l_inf_proxy_hidden_dim=l_inf_proxy_hidden_dim,
        l_inf_proxy_series=l_inf_proxy_series,
        l_inf_proxy_source=l_inf_proxy_source,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        force_resample=args.force_resample,
    )

    for row in rows:
        active_targets = CURVE_TARGETS[args.curve]
        parts = [f"hidden_dim={row['hidden_dim']:.0f}"]
        if "mi" in active_targets:
            parts.append(f"n*_I={row['n_star_i']:.3f}")
        if "l" in active_targets:
            parts.append(f"n*_L={row['n_star_l']:.3f}")
        if "h" in active_targets:
            parts.append(f"n*_H={row['n_star_h']:.3f}")
        print(", ".join(parts))

    out_path = (
        args.output if args.output is not None else f"results/n_star_{args.group}.png"
    )
    if args.curve == "mi":
        title = (
            "Critical n*_I "
            f"(group={args.group}, mi_estimator={args.mi_estimator}, "
            f"mi_ref={mi_ref_label})"
        )
    elif args.curve == "l":
        title = f"Critical n*_L (group={args.group}, l_ref={l_ref_label})"
    elif args.curve == "lmi":
        title = (
            f"Critical n* (group={args.group}, mi_estimator={args.mi_estimator}, "
            f"mi_ref={mi_ref_label}, l_ref={l_ref_label})"
        )
    elif args.curve == "h":
        title = f"Critical n*_H (group={args.group}, h_ref={h_ref_label})"
    else:
        title = (
            f"Critical n* (group={args.group}, mi_estimator={args.mi_estimator}, "
            f"mi_ref={mi_ref_label}, "
            f"l_ref={l_ref_label}, h_ref={h_ref_label})"
        )
    _plot_n_star(
        rows,
        out_path,
        title,
        args.curve,
        fit=bool(args.fit),
    )


if __name__ == "__main__":
    main()

"""Plot critical n* curves vs hidden_dim using sampled MI and L_n."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import termios
import tty
from types import SimpleNamespace

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
    _compute_bipartite_mi_from_sampled_q,
    _compute_log_q_y_means,
    _download_checkpoint_artifact,
    _find_reusable_complete_cache,
    _load_checkpoint,
    _normalize_params_for_step,
    _sample_cache_key,
    _sample_cache_paths,
    _sample_sequences,
    _save_log_q_y_mean_cache,
    _save_sample_cache,
    _select_cached_n_values,
)

WANDB_PROJECT = "tarunadvaith-/scaling"
DEFAULT_REF_COEF = 1.0
DEFAULT_REF_POWER = -0.5
DEFAULT_REF_OFFSET = 0.0
FIT_NMAX = 10
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


def _compute_sampled_mi_series(
    *,
    run: wandb.apis.public.Run,
    api: wandb.Api,
    n_values: list[int],
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> dict[int, float]:
    cfg = run.config or {}
    hidden_dim = int(cfg["hidden_dim"])

    bos_token_id = int(cfg.get("bos_token_id", 0))
    sample_key = _sample_cache_key(
        seq_len=n_values[-1],
        num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
    )
    sample_cache_path, log_q_y_cache_path = _sample_cache_paths(
        cache_dir,
        run.id,
        sample_key,
    )

    if not force_resample:
        reusable_cache = _find_reusable_complete_cache(
            cache_dir=cache_dir,
            run_id=run.id,
            seq_len=n_values[-1],
            target_num_samples=num_samples,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            n_values=n_values,
        )
        if reusable_cache is None:
            raise RuntimeError(
                "No complete cached sampled MI found for "
                f"hidden_dim={hidden_dim}. Re-run with --force-resample to regenerate."
            )
        (
            reusable_sample_logps,
            reusable_log_q_y_means_by_n,
            reusable_num_samples,
            reusable_sample_cache_path,
            _,
        ) = reusable_cache
        available_n_values = _select_cached_n_values(
            n_values,
            reusable_log_q_y_means_by_n,
            reusable_sample_logps,
        )
        if not available_n_values:
            raise RuntimeError(
                "Cached sampled artifacts have no usable N values for "
                f"hidden_dim={hidden_dim}. Re-run with --force-resample to regenerate."
            )
        if len(available_n_values) < len(n_values):
            print(
                "Using cached sampled MI subset for hidden_dim="
                f"{hidden_dim}: {len(available_n_values)}/{len(n_values)} "
                "N values available"
            )
        print(
            "Using cached sampled MI for hidden_dim="
            f"{hidden_dim} from {os.path.basename(reusable_sample_cache_path)} "
            f"(num_samples={reusable_num_samples}, requested={num_samples})"
        )
        return _compute_bipartite_mi_from_sampled_q(
            sample_logps=reusable_sample_logps,
            n_values=available_n_values,
            log_q_y_means_by_n=reusable_log_q_y_means_by_n,
        )

    print(
        "Force resample enabled for hidden_dim="
        f"{hidden_dim}; regenerating sampled cache"
    )
    if os.path.exists(sample_cache_path):
        os.remove(sample_cache_path)
    if os.path.exists(log_q_y_cache_path):
        os.remove(log_q_y_cache_path)

    import jax

    from models.lstm import LSTMLanguageModel
    from training.trainer import create_train_state

    ckpt_path = _download_checkpoint_artifact(run.id, api, cache_dir)
    model = LSTMLanguageModel(
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        vocab_size=int(cfg["vocab_size"]),
    )
    rng = jax.random.PRNGKey(0)
    state_cfg = SimpleNamespace(
        batch_size=int(cfg.get("batch_size", 1)),
        seq_len=int(cfg.get("seq_len", n_values[-1])),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )
    state = create_train_state(model, state_cfg, rng)
    state, restored = _load_checkpoint(ckpt_path, state)
    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    sample_params = _normalize_params_for_step(
        state.params,
        int(cfg["num_layers"]),
    )
    samples, sample_logps = _sample_sequences(
        model=model,
        params=sample_params,
        seq_len=n_values[-1],
        num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        rng=rng,
        progress_desc=f"Sampling d_h={hidden_dim}",
    )
    _save_sample_cache(sample_cache_path, samples, sample_logps)

    log_q_y_means_by_n = _compute_log_q_y_means(
        samples=samples,
        apply_fn=state.apply_fn,
        params=sample_params,
        n_values=n_values,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        progress_desc_prefix=f"Scoring y d_h={hidden_dim}",
    )
    _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)

    return _compute_bipartite_mi_from_sampled_q(
        sample_logps=sample_logps,
        n_values=n_values,
        log_q_y_means_by_n=log_q_y_means_by_n,
    )


def _compute_n_star_rows(
    *,
    runs: list[wandb.apis.public.Run],
    api: wandb.Api,
    n_values: list[int],
    curve: str,
    mi_ref_coef: float,
    mi_ref_power: float,
    mi_ref_offset: float,
    l_ref_coef: float,
    l_ref_power: float,
    l_ref_offset: float,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> tuple[list[dict[str, float]], str, str]:
    mi_by_hidden_dim: dict[int, dict[int, float]] = {}
    l_by_hidden_dim: dict[int, dict[int, float]] = {}

    for run in tqdm(runs, desc="Runs", unit="run"):
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])

        if curve in {"both", "mi"}:
            mi_series = _compute_sampled_mi_series(
                run=run,
                api=api,
                n_values=n_values,
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

        if curve in {"both", "l"}:
            l_series = _extract_combined_ngram_losses(run)
            if not l_series:
                raise RuntimeError(
                    "No combined n-gram losses available for "
                    f"run '{run.name}' (hidden_dim={hidden_dim})"
                )
            l_by_hidden_dim[hidden_dim] = l_series

    if curve == "both":
        selected_hidden_dims = sorted(
            set(mi_by_hidden_dim.keys()) & set(l_by_hidden_dim.keys())
        )
        if not selected_hidden_dims:
            raise RuntimeError("No hidden_dim has both sampled MI and L_n series")
    elif curve == "mi":
        selected_hidden_dims = sorted(mi_by_hidden_dim.keys())
        if not selected_hidden_dims:
            raise RuntimeError("No hidden_dim has sampled MI series")
    else:
        selected_hidden_dims = sorted(l_by_hidden_dim.keys())
        if not selected_hidden_dims:
            raise RuntimeError("No hidden_dim has L_n series")

    mi_reference_label = (
        f"power_law(coef={mi_ref_coef}, power={mi_ref_power}, offset={mi_ref_offset})"
        if curve in {"both", "mi"}
        else "disabled"
    )
    l_reference_label = (
        f"power_law(coef={l_ref_coef}, power={l_ref_power}, offset={l_ref_offset})"
        if curve in {"both", "l"}
        else "disabled"
    )

    rows: list[dict[str, float]] = []
    for hidden_dim in tqdm(selected_hidden_dims, desc="Computing n*", unit="run"):
        row: dict[str, float] = {"hidden_dim": float(hidden_dim)}

        if curve in {"both", "mi"}:
            mi_inf = _fit_asymptote(mi_by_hidden_dim[hidden_dim])
            n_star_i = _solve_power_law_n(
                target=mi_inf,
                coef=mi_ref_coef,
                power=mi_ref_power,
                offset=mi_ref_offset,
            )
            row["mi_inf"] = mi_inf
            row["n_star_i"] = float(n_star_i)

        if curve in {"both", "l"}:
            l_inf = _fit_asymptote(l_by_hidden_dim[hidden_dim])
            n_star_l = _solve_power_law_n(
                target=l_inf,
                coef=l_ref_coef,
                power=l_ref_power,
                offset=l_ref_offset,
            )
            row["l_inf"] = l_inf
            row["n_star_l"] = float(n_star_l)

        rows.append(row)

    return rows, mi_reference_label, l_reference_label


def _plot_n_star(
    rows: list[dict[str, float]],
    out_path: str,
    title: str,
    curve: str,
) -> None:
    if not rows:
        raise RuntimeError("No rows to plot")

    hidden_dims = np.array([row["hidden_dim"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    if curve in {"both", "mi"}:
        n_star_i = np.array([row["n_star_i"] for row in rows], dtype=float)
        ax.plot(hidden_dims, n_star_i, marker="o", linestyle="-", label=r"$n^*_{MI}$")
    if curve in {"both", "l"}:
        n_star_l = np.array([row["n_star_l"] for row in rows], dtype=float)
        ax.plot(hidden_dims, n_star_l, marker="s", linestyle="--", label=r"$n^*_{L_n}$")
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
        choices=["both", "mi", "l"],
        default="both",
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
        "--mi-ref-coef",
        type=float,
        default=2.41,
        help="Power-law coefficient for MI reference",
    )
    parser.add_argument(
        "--mi-ref-power",
        type=float,
        default=0.366,
        help="Power-law exponent for MI reference",
    )
    parser.add_argument(
        "--mi-ref-offset",
        type=float,
        default=0.0,
        help="Power-law offset for MI reference",
    )
    parser.add_argument(
        "--l-ref-coef",
        type=float,
        default=3.86,
        help="Power-law coefficient for L reference",
    )
    parser.add_argument(
        "--l-ref-power",
        type=float,
        default=-0.950,
        help="Power-law exponent for L reference",
    )
    parser.add_argument(
        "--l-ref-offset",
        type=float,
        default=2.98,
        help="Power-law offset for L reference",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="Maximum N to include in sampled MI curve",
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
        default=1000,
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

    api = wandb.Api()
    runs = _resolve_group_runs(api, args.group)
    runs = [
        run
        for run in runs
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

    rows, mi_ref_label, l_ref_label = _compute_n_star_rows(
        runs=runs,
        api=api,
        n_values=n_values,
        curve=args.curve,
        mi_ref_coef=args.mi_ref_coef,
        mi_ref_power=args.mi_ref_power,
        mi_ref_offset=args.mi_ref_offset,
        l_ref_coef=args.l_ref_coef,
        l_ref_power=args.l_ref_power,
        l_ref_offset=args.l_ref_offset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        force_resample=args.force_resample,
    )

    for row in rows:
        if args.curve == "both":
            print(
                "hidden_dim={hidden_dim:.0f}, "
                "n*_I={n_star_i:.3f}, n*_L={n_star_l:.3f}".format(**row)
            )
        elif args.curve == "mi":
            print("hidden_dim={hidden_dim:.0f}, n*_I={n_star_i:.3f}".format(**row))
        else:
            print("hidden_dim={hidden_dim:.0f}, n*_L={n_star_l:.3f}".format(**row))

    out_path = (
        args.output if args.output is not None else f"results/n_star_{args.group}.png"
    )
    if args.curve == "both":
        title = (
            f"Critical n* (group={args.group}, mi_ref={mi_ref_label}, "
            f"l_ref={l_ref_label})"
        )
    elif args.curve == "mi":
        title = f"Critical n*_I (group={args.group}, mi_ref={mi_ref_label})"
    else:
        title = f"Critical n*_L (group={args.group}, l_ref={l_ref_label})"
    _plot_n_star(rows, out_path, title, args.curve)


if __name__ == "__main__":
    main()

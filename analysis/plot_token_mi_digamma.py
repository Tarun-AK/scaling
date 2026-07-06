"""Plot token mutual information vs separation using digamma entropy estimator.

Uses cached samples produced by analysis/plot_bipartite_mi.py.
For each pair (0, j), where n = j - 1 is the number of tokens in between the
first token and the target token, estimate

  I(X;Y) = H(X) + H(Y) - H(X,Y)

with

  H = ln N - (1/N) * sum_i n_i * digamma(n_i).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import termios
import tty

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma
from tqdm import tqdm

import wandb
from analysis.plot_bipartite_mi import _filter_runs_by_hidden_dim, _resolve_group_runs
from analysis.plot_group_labels import distinct_group_labels

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


def _parse_model_sample_cache(filename: str) -> tuple[int, int, int, int] | None:
    match = re.match(
        r"^samples_seq(?P<seq>\d+)_num(?P<num>\d+)_samplebs(?P<bs>\d+)_bos(?P<bos>\d+)\.npz$",
        filename,
    )
    if match is None:
        return None
    return (
        int(match.group("seq")),
        int(match.group("num")),
        int(match.group("bs")),
        int(match.group("bos")),
    )


def _parse_data_sample_cache(filename: str) -> tuple[int, int, int] | None:
    match = re.match(
        r"^data_samples_data_(?:validation|train|test)_seq(?P<seq>\d+)_num(?P<num>all|\d+)_bos(?P<bos>\d+)(?:_seed\d+)?\.npz$",
        filename,
    )
    if match is None:
        return None
    num_tag = match.group("num")
    num_score = 10**12 if num_tag == "all" else int(num_tag)
    return (int(match.group("seq")), int(num_score), int(match.group("bos")))


def _choose_sample_cache(
    *,
    cache_dir: str,
    run_id: str,
    sample_source: str,
    parser_fn,
) -> str:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    if not os.path.isdir(run_cache_dir):
        raise RuntimeError(f"Missing run cache directory: {run_cache_dir}")

    entries: list[tuple[tuple[int, ...], str]] = []
    for filename in os.listdir(run_cache_dir):
        parsed = parser_fn(filename)
        if parsed is None:
            continue
        entries.append((parsed, os.path.join(run_cache_dir, filename)))

    if not entries:
        raise RuntimeError(
            f"No {sample_source} sample cache found for run_id={run_id} in {run_cache_dir}"
        )

    entries.sort(key=lambda item: item[0], reverse=True)
    return entries[0][1]


def _choose_model_sample_cache(*, cache_dir: str, run_id: str) -> str:
    return _choose_sample_cache(
        cache_dir=cache_dir,
        run_id=run_id,
        sample_source="model",
        parser_fn=_parse_model_sample_cache,
    )


def _choose_data_sample_cache(*, cache_dir: str, run_id: str) -> str:
    return _choose_sample_cache(
        cache_dir=cache_dir,
        run_id=run_id,
        sample_source="data",
        parser_fn=_parse_data_sample_cache,
    )


def _load_samples_only(sample_cache_path: str) -> np.ndarray:
    with np.load(sample_cache_path) as data:
        if "samples" not in data:
            raise RuntimeError(f"No 'samples' array in cache: {sample_cache_path}")
        samples = np.array(data["samples"], dtype=np.int32)
    if samples.ndim != 2:
        raise RuntimeError(
            f"Expected 2D samples array, got shape={samples.shape} in {sample_cache_path}"
        )
    return samples


def _first_token_pairs(seq_len: int) -> list[tuple[int, int, int]]:
    if seq_len < 2:
        return []
    return [(j - 1, 0, j) for j in range(1, seq_len)]


def _sval_cache_path(sample_cache_path: str) -> str:
    sample_file = os.path.basename(sample_cache_path)
    sample_id = sample_file[:-4]
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", sample_id)
    return os.path.join(
        os.path.dirname(sample_cache_path),
        f"token_mi_digamma_{safe_id}_first_anchor_inbetween.npz",
    )


def _load_curve_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path) as data:
        if "n_values" not in data or "mi_values" not in data:
            return None
        n_values = np.array(data["n_values"], dtype=np.int32)
        mi_values = np.array(data["mi_values"], dtype=np.float64)
    if n_values.shape != mi_values.shape:
        return None
    return n_values, mi_values


def _save_curve_cache(cache_path: str, n_values: np.ndarray, mi_values: np.ndarray) -> None:
    np.savez_compressed(
        cache_path,
        n_values=n_values.astype(np.int32),
        mi_values=mi_values.astype(np.float64),
    )


def _entropy_digamma_from_counts(counts: np.ndarray) -> float:
    counts_f = np.asarray(counts, dtype=np.float64)
    counts_f = counts_f[counts_f > 0]
    if counts_f.size == 0:
        return 0.0
    n_total = float(np.sum(counts_f))
    return float(np.log(n_total) - (1.0 / n_total) * np.sum(counts_f * digamma(counts_f)))


def _token_pair_mi_digamma(
    left_tokens: np.ndarray,
    right_tokens: np.ndarray,
    vocab_size: int,
) -> float:
    left_counts = np.bincount(left_tokens, minlength=vocab_size)
    right_counts = np.bincount(right_tokens, minlength=vocab_size)
    h_x = _entropy_digamma_from_counts(left_counts)
    h_y = _entropy_digamma_from_counts(right_counts)

    pair_ids = left_tokens.astype(np.int64) * int(vocab_size) + right_tokens.astype(np.int64)
    _, joint_counts = np.unique(pair_ids, return_counts=True)
    h_xy = _entropy_digamma_from_counts(joint_counts)
    return float(h_x + h_y - h_xy)


def _compute_curve_for_run(
    *,
    sample_cache_path: str,
    vocab_size: int,
    max_n: int | None,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _sval_cache_path(sample_cache_path)
    cached_n_values = np.array([], dtype=np.int32)
    cached_mi_values = np.array([], dtype=np.float64)
    if not force_recompute:
        cached = _load_curve_cache(cache_path)
        if cached is not None:
            cached_n_values, cached_mi_values = cached

    if force_recompute:
        cached_lookup: dict[int, float] = {}
    else:
        cached_lookup = {
            int(n): float(v)
            for n, v in zip(cached_n_values.tolist(), cached_mi_values.tolist())
        }

    samples = _load_samples_only(sample_cache_path)
    seq_len = int(samples.shape[1])
    pairs = _first_token_pairs(seq_len)
    if max_n is not None:
        pairs = [item for item in pairs if int(item[0]) <= int(max_n)]
    if not pairs:
        raise RuntimeError("No valid first-token pairs for requested settings")

    missing_pairs = [item for item in pairs if int(item[0]) not in cached_lookup]
    for n_between, i, j in tqdm(missing_pairs, desc="n", unit="pair", leave=False):
        left = np.asarray(samples[:, i], dtype=np.int32)
        right = np.asarray(samples[:, j], dtype=np.int32)
        cached_lookup[int(n_between)] = _token_pair_mi_digamma(
            left_tokens=left,
            right_tokens=right,
            vocab_size=vocab_size,
        )

    n_values = np.array(sorted(int(item[0]) for item in pairs), dtype=np.int32)
    mi_values = np.array([float(cached_lookup[int(n)]) for n in n_values], dtype=np.float64)

    n_values_all = np.array(sorted(cached_lookup.keys()), dtype=np.int32)
    mi_values_all = np.array(
        [float(cached_lookup[int(n)]) for n in n_values_all],
        dtype=np.float64,
    )
    _save_curve_cache(cache_path, n_values_all, mi_values_all)
    return n_values, mi_values


def _plot_curves(
    curves_by_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    out_path: str,
    title: str,
) -> None:
    if not curves_by_series:
        raise RuntimeError("No curves to plot")

    fig, ax = plt.subplots(figsize=(8, 6))
    series_keys = sorted(curves_by_series)
    hidden_dims = sorted({hidden_dim for _, hidden_dim in series_keys})
    groups = sorted({group for group, _ in series_keys})
    group_labels = distinct_group_labels(groups) if len(groups) > 1 else {}
    show_group_name = len(groups) > 1
    cmap = plt.get_cmap("viridis")
    if len(hidden_dims) > 1:
        norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))
    else:
        norm = plt.Normalize(vmin=hidden_dims[0], vmax=hidden_dims[0] + 1)

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_group = {
        group: marker_cycle[idx % len(marker_cycle)] for idx, group in enumerate(groups)
    }

    for group_name, hidden_dim in series_keys:
        n_values, mi_values = curves_by_series[(group_name, hidden_dim)]
        if n_values.size == 0:
            continue
        mask = (n_values > 0) & (mi_values > 0)
        if not np.any(mask):
            continue
        ax.plot(
            n_values[mask],
            mi_values[mask],
            linewidth=1.5,
            color=cmap(norm(hidden_dim)),
            marker=marker_by_group[group_name],
            label=(
                f"d_h={hidden_dim} [{group_labels.get(group_name, group_name)}]"
                if show_group_name
                else f"d_h={hidden_dim}"
            ),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (tokens in between)")
    ax.set_ylabel(r"$\hat{I}(X_0; X_{n+1})$")
    ax.set_title(title)
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

    if len(hidden_dims) > 1:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$d_h$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, nargs="+", required=True)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Optional max n (tokens in between) to compute/plot",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    args = parser.parse_args()

    api = wandb.Api()
    groups = list(dict.fromkeys(args.group))
    curves_by_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for group_name in groups:
        runs = _resolve_group_runs(api, group_name)
        runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, group_name)
        runs = [
            run
            for run in runs
            if int((run.config or {}).get("hidden_dim", -1)) != 128
        ]
        if not runs:
            raise RuntimeError(
                f"No runs left after hidden_dim=128 filter for group={group_name}"
            )

        for run in tqdm(runs, desc=f"Runs[{group_name}]", unit="run"):
            cfg = run.config or {}
            hidden_dim = int(cfg["hidden_dim"])
            vocab_size = int(cfg["vocab_size"])
            try:
                sample_cache_path = _choose_model_sample_cache(
                    cache_dir=args.cache_dir,
                    run_id=run.id,
                )
            except RuntimeError as exc:
                print(
                    f"skip group={group_name} hidden_dim={hidden_dim}: {exc}"
                )
                continue
            print(
                f"group={group_name} hidden_dim={hidden_dim}: "
                f"using model sample cache {os.path.basename(sample_cache_path)}"
            )
            curves_by_series[(group_name, hidden_dim)] = _compute_curve_for_run(
                sample_cache_path=sample_cache_path,
                vocab_size=vocab_size,
                max_n=args.max_n,
                force_recompute=bool(args.force_recompute),
            )

    if not curves_by_series:
        raise RuntimeError("No model sample caches found for requested groups")

    out_path = (
        args.output
        if args.output is not None
        else f"results/token_mi_digamma_{'_'.join(groups)}.png"
    )
    _plot_curves(
        curves_by_series,
        out_path,
        title=(f"Token MI (digamma estimator), groups={','.join(groups)}"),
    )


if __name__ == "__main__":
    main()

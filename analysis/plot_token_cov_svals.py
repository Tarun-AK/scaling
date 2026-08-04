"""Plot largest singular value of token-pair covariance vs separation.

Uses cached samples produced by analysis/plot_bipartite_mi.py.
For each pair (0, j), where n = j - 1 is the number of tokens in between the
first token and the target token, compute the covariance operator

  C_n = P(x_i, x_j) - P(x_i) P(x_j)^T

over vocabulary tokens and plot sigma_max(C_n) vs n.
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds
from tqdm import tqdm

import wandb
from analysis.plot_bipartite_mi import _filter_runs_by_hidden_dim, _resolve_group_runs
from analysis.plot_group_labels import distinct_group_labels
from data.dataset import load_splits_as_arrays

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
    return (
        int(match.group("seq")),
        int(num_score),
        int(match.group("bos")),
    )


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

    # Highest seq_len first, then largest sample count.
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


def _load_data_split_samples(
    cfg: dict,
    split: str,
) -> np.ndarray:
    dataset_name = str(cfg["dataset_name"])
    dataset_config = cfg.get("dataset_config")
    dataset_path = cfg.get("dataset_path")
    seq_len = int(cfg["seq_len"])
    vocab_size = int(cfg["vocab_size"])
    cache_dir = str(cfg.get("cache_dir", "data/cache"))
    require_cache = bool(cfg.get("require_cached_data", True))
    tokenize_batch_size = int(cfg.get("tokenize_batch_size", 32))
    tokenizer_path = str(cfg.get("tokenizer_path", "data/tokenizer/tokenizer.json"))

    train_np, val_np, test_np = load_splits_as_arrays(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        require_cache=require_cache,
        tokenize_batch_size=tokenize_batch_size,
        tokenizer_path=tokenizer_path,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
    )

    if split == "train":
        return np.asarray(train_np, dtype=np.int32)
    if split == "validation":
        return np.asarray(val_np, dtype=np.int32)
    if split == "test":
        return np.asarray(test_np, dtype=np.int32)
    if split == "all":
        return np.asarray(
            np.concatenate([train_np, val_np, test_np], axis=0),
            dtype=np.int32,
        )
    raise RuntimeError(f"Unsupported data split: {split}")


def _load_data_split_samples_from_args(args: argparse.Namespace) -> np.ndarray:
    cfg = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_path": args.dataset_path,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "cache_dir": args.dataset_cache_dir,
        "require_cached_data": args.require_cached_data,
        "tokenize_batch_size": args.tokenize_batch_size,
        "tokenizer_path": args.tokenizer_path,
    }
    return _load_data_split_samples(cfg, args.data_split)


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


def _cov_operator(
    left_tokens: np.ndarray,
    right_tokens: np.ndarray,
    vocab_size: int,
) -> LinearOperator:
    num_samples = int(left_tokens.shape[0])
    if num_samples < 1:
        raise RuntimeError("Need at least one sample to build covariance operator")

    weights = np.full((num_samples,), 1.0 / float(num_samples), dtype=np.float64)
    joint = csr_matrix(
        (weights, (left_tokens, right_tokens)),
        shape=(int(vocab_size), int(vocab_size)),
        dtype=np.float64,
    )
    p_left = np.bincount(left_tokens, minlength=vocab_size).astype(np.float64)
    p_right = np.bincount(right_tokens, minlength=vocab_size).astype(np.float64)
    p_left /= float(num_samples)
    p_right /= float(num_samples)

    def _matvec(x: np.ndarray) -> np.ndarray:
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        return np.asarray(joint.dot(x_vec)).reshape(-1) - p_left * float(
            np.dot(p_right, x_vec)
        )

    def _rmatvec(y: np.ndarray) -> np.ndarray:
        y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
        return np.asarray(joint.transpose().dot(y_vec)).reshape(-1) - p_right * float(
            np.dot(p_left, y_vec)
        )

    return LinearOperator(
        shape=(int(vocab_size), int(vocab_size)),
        matvec=_matvec,
        rmatvec=_rmatvec,
        dtype=np.float64,
    )


def _largest_singular_value(
    left_tokens: np.ndarray,
    right_tokens: np.ndarray,
    vocab_size: int,
    tol: float,
    maxiter: int,
) -> float:
    op = _cov_operator(left_tokens, right_tokens, vocab_size)
    sval = svds(
        op,
        k=1,
        return_singular_vectors=False,
        which="LM",
        tol=float(tol),
        maxiter=int(maxiter),
    )
    if sval.size == 0:
        return 0.0
    return float(abs(sval[-1]))


def _first_token_pairs(seq_len: int) -> list[tuple[int, int, int]]:
    if seq_len < 2:
        return []
    out: list[tuple[int, int, int]] = []
    i = 0
    for j in range(1, seq_len):
        n_between = j - i - 1
        out.append((n_between, i, j))
    return out


def _sval_cache_path(
    sample_cache_path: str,
) -> str:
    sample_file = os.path.basename(sample_cache_path)
    sample_id = sample_file[:-4]
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", sample_id)
    return os.path.join(
        os.path.dirname(sample_cache_path),
        f"token_cov_sval_{safe_id}_first_anchor_inbetween.npz",
    )


def _load_sval_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path) as data:
        if "n_values" not in data or "svals" not in data:
            return None
        n_values = np.array(data["n_values"], dtype=np.int32)
        svals = np.array(data["svals"], dtype=np.float64)
    if n_values.shape != svals.shape:
        return None
    return n_values, svals


def _save_sval_cache(cache_path: str, n_values: np.ndarray, svals: np.ndarray) -> None:
    np.savez_compressed(cache_path, n_values=n_values.astype(np.int32), svals=svals)


def _compute_curve_for_run(
    *,
    sample_cache_path: str,
    vocab_size: int,
    max_n: int | None,
    tol: float,
    maxiter: int,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _sval_cache_path(sample_cache_path)
    cached_n_values = np.array([], dtype=np.int32)
    cached_svals = np.array([], dtype=np.float64)
    if not force_recompute:
        cached = _load_sval_cache(cache_path)
        if cached is not None:
            cached_n_values, cached_svals = cached

    if force_recompute:
        cached_lookup: dict[int, float] = {}
    else:
        cached_lookup = {
            int(n): float(v) for n, v in zip(cached_n_values.tolist(), cached_svals.tolist())
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
        cached_lookup[int(n_between)] = _largest_singular_value(
            left_tokens=left,
            right_tokens=right,
            vocab_size=vocab_size,
            tol=tol,
            maxiter=maxiter,
        )

    n_values = np.array(sorted(int(item[0]) for item in pairs), dtype=np.int32)
    svals = np.array([float(cached_lookup[int(n)]) for n in n_values], dtype=np.float64)

    n_values_all = np.array(sorted(cached_lookup.keys()), dtype=np.int32)
    svals_all = np.array([float(cached_lookup[int(n)]) for n in n_values_all], dtype=np.float64)
    _save_sval_cache(cache_path, n_values_all, svals_all)
    return n_values, svals


def _compute_data_curve_for_run(
    *,
    cfg: dict,
    data_split: str,
    max_n: int | None,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = _load_data_split_samples(cfg, data_split)
    vocab_size = int(cfg["vocab_size"])
    return _compute_data_curve_from_samples(
        samples=samples,
        vocab_size=vocab_size,
        max_n=max_n,
        tol=tol,
        maxiter=maxiter,
        data_split=data_split,
    )


def _compute_data_curve_from_samples(
    *,
    samples: np.ndarray,
    vocab_size: int,
    max_n: int | None,
    tol: float,
    maxiter: int,
    data_split: str,
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = int(samples.shape[1])
    pairs = _first_token_pairs(seq_len)
    if max_n is not None:
        pairs = [item for item in pairs if int(item[0]) <= int(max_n)]
    if not pairs:
        raise RuntimeError("No valid first-token pairs for requested settings")

    cache_lookup: dict[int, float] = {}
    for n_between, i, j in tqdm(
        pairs, desc=f"data[{data_split}] n", unit="pair", leave=False
    ):
        left = np.asarray(samples[:, i], dtype=np.int32)
        right = np.asarray(samples[:, j], dtype=np.int32)
        cache_lookup[int(n_between)] = _largest_singular_value(
            left_tokens=left,
            right_tokens=right,
            vocab_size=vocab_size,
            tol=tol,
            maxiter=maxiter,
        )

    n_values = np.array(sorted(int(item[0]) for item in pairs), dtype=np.int32)
    svals = np.array([float(cache_lookup[int(n)]) for n in n_values], dtype=np.float64)
    return n_values, svals


def _plot_curves(
    curves_by_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    data_curve_by_group: dict[str, tuple[np.ndarray, np.ndarray]],
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
        n_values, svals = curves_by_series[(group_name, hidden_dim)]
        if n_values.size == 0:
            continue
        mask = n_values > 0
        if not np.any(mask):
            continue
        ax.plot(
            n_values[mask],
            svals[mask],
            linewidth=1.5,
            color=cmap(norm(hidden_dim)),
            marker=marker_by_group[group_name],
            label=(
                f"d_h={hidden_dim} [{group_labels.get(group_name, group_name)}]"
                if show_group_name
                else f"d_h={hidden_dim}"
            ),
        )

    for group_name, data_curve in sorted(data_curve_by_group.items()):
        n_values_data, svals_data = data_curve
        if n_values_data.size == 0:
            continue
        mask_data = n_values_data > 0
        if np.any(mask_data):
                ax.plot(
                    n_values_data[mask_data],
                    svals_data[mask_data],
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    label=(
                        f"data [{group_labels.get(group_name, group_name)}]"
                        if show_group_name
                        else "data"
                    ),
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (tokens in between)")
    ax.set_ylabel(r"$\sigma_{\max}(C_n)$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
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
    parser.add_argument("--group", type=str, nargs="+", default=None)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Plot only a data curve using the training-script dataset loader",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="openwebtext",
        help="Dataset name for --data-only mode",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config for --data-only mode",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="openwebtext",
        help="Dataset path for --data-only mode",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default="data/openwebtext_cache",
        help="Cache dir for tokenized dataset splits in --data-only mode",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length for --data-only mode",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size for --data-only mode",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="openwebtext/tokenizer/tokenizer.json",
        help="Tokenizer path for --data-only mode",
    )
    parser.add_argument(
        "--require-cached-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require cached tokenized splits in --data-only mode",
    )
    parser.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=32,
        help="Tokenization batch size for --data-only mode",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["validation", "train", "test", "all"],
        default="validation",
        help="Cached dataset split to use for the data overlay",
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
    parser.add_argument("--svd-tol", type=float, default=1e-6)
    parser.add_argument("--svd-maxiter", type=int, default=5000)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    args = parser.parse_args()

    if args.data_only:
        data_samples = _load_data_split_samples_from_args(args)
        data_curve = _compute_data_curve_from_samples(
            samples=data_samples,
            vocab_size=args.vocab_size,
            data_split=args.data_split,
            max_n=args.max_n,
            tol=args.svd_tol,
            maxiter=args.svd_maxiter,
        )
        out_path = args.output or f"results/token_cov_svals_data_{args.data_split}.png"
        _plot_curves(
            {},
            {"data": data_curve},
            out_path,
            title=f"Largest singular value of token covariance (data={args.data_split})",
        )
        return

    if not args.group:
        raise RuntimeError("--group is required unless --data-only is set")

    api = wandb.Api()
    groups = list(dict.fromkeys(args.group))
    curves_by_series: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    data_curve_by_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
                f"No runs left after temporary hidden_dim=128 filter for group={group_name}"
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
                tol=args.svd_tol,
                maxiter=args.svd_maxiter,
                force_recompute=bool(args.force_recompute),
            )

        data_curve = None
        for run in runs:
            cfg = run.config or {}
            try:
                data_curve = _compute_data_curve_for_run(
                    cfg=cfg,
                    data_split=args.data_split,
                    max_n=args.max_n,
                    tol=args.svd_tol,
                    maxiter=args.svd_maxiter,
                )
                print(
                    f"group={group_name}: using dataset split '{args.data_split}' for overlay"
                )
                break
            except Exception as exc:
                print(
                    f"skip data overlay for group={group_name} hidden_dim={int(cfg.get('hidden_dim', -1))}: {exc}"
                )
                continue

        if data_curve is None:
            print(f"skip data overlay for group={group_name}: no data sample cache found")
        else:
            data_curve_by_group[group_name] = data_curve

    if not curves_by_series:
        raise RuntimeError("No model sample caches found for requested groups")

    out_path = (
        args.output
        if args.output is not None
        else f"results/token_cov_svals_{'_'.join(groups)}.png"
    )
    _plot_curves(
        curves_by_series,
        data_curve_by_group,
        out_path,
        title=(
            "Largest singular value of token covariance"
            f" (groups={','.join(groups)})"
        ),
    )


if __name__ == "__main__":
    main()

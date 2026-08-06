"""Per-token-checkpoint L_n, read from the MI cache.

L_n = -E[log p(x_n | x_<n)], as defined in training/metrics.py.

The runs' logged `*/ngram_N` metrics only exist at step 0 and end of epoch, so
they carry no per-checkpoint dimension. The MI cache, however, already stores
per-position log p(x_n | x_<n) for every scored sequence of every token
checkpoint -- so L_n per checkpoint is a mean over sequences of that array, at
no GPU cost for anything already scored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import termios
import tty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb


def _show_image(path: str) -> None:
    """Display inline via kitty, then clear on keypress. Same as the other
    plotting scripts -- every one of them shows the figure after saving."""
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


def ngram_losses_by_token_checkpoint(
    group: str,
    *,
    hidden_dims: list[int] | None = None,
    cache_dir: str = "checkpoints/bipartite_mi_cache",
    data_split: str = "validation",
    batch_size: int = 32,
    force_resample: bool = False,
) -> pd.DataFrame:
    """L_n for every (hidden_dim, token checkpoint) in a group.

    Returns columns: hidden_dim, tokens, n, loss. Anything not already scored
    is computed through the usual estimator, which populates the same cache the
    MI plots use, so the work is shared rather than duplicated.
    """
    from analysis.plot_bipartite_mi import (
        DEFAULT_N_VALUES,
        _cache_namespace,
        _compute_lstm_direct_mi_for_run_from_data,
        _compute_with_cache_fill,
        _data_cache_key,
        _data_cache_paths,
        _filter_runs_by_hidden_dim,
        _list_token_checkpoints,
        _load_sample_logps,
        _resolve_group_runs,
        token_checkpoint,
    )

    api = wandb.Api()
    runs = _filter_runs_by_hidden_dim(
        _resolve_group_runs(api, group), hidden_dims, group
    )
    if not runs:
        raise RuntimeError(f"No runs found for group='{group}'")

    rows: list[dict] = []
    for run in runs:
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])
        bos_token_id = int(cfg.get("bos_token_id", 0))
        n_values = [n for n in DEFAULT_N_VALUES if n <= int(cfg["seq_len"])]
        if not n_values:
            continue
        milestones = _list_token_checkpoints(run)
        if not milestones:
            print(f"  {run.id} hd={hidden_dim}: no token checkpoints; skipping")
            continue
        data_key = _data_cache_key(
            split=data_split,
            seq_len=int(n_values[-1]),
            num_samples=None,
            bos_token_id=bos_token_id,
            seed=None,
        )
        print(f"  {run.id} hd={hidden_dim}: {len(milestones)} token checkpoints")
        for tokens in milestones:
            checkpoint = token_checkpoint(tokens)
            sample_path, _ = _data_cache_paths(
                cache_dir, _cache_namespace(run.id, checkpoint), data_key
            )
            if force_resample or _load_sample_logps(sample_path) is None:
                _compute_with_cache_fill(
                    lambda force, _ckpt=checkpoint, _run=run, _hd=hidden_dim, _nv=n_values: (
                        _compute_lstm_direct_mi_for_run_from_data(
                            _run,
                            api,
                            _hd,
                            _nv,
                            batch_size=batch_size,
                            cache_dir=cache_dir,
                            force_resample=force,
                            data_split=data_split,
                            checkpoint=_ckpt,
                        )
                    ),
                    force_resample=force_resample,
                    fill_missing=True,
                    desc=f"ngram/data hidden_dim={hidden_dim} checkpoint={checkpoint}",
                )
            sample_logps = _load_sample_logps(sample_path)
            if sample_logps is None:
                print(f"    no scored logps for {checkpoint}; skipping")
                continue
            losses = -sample_logps.mean(axis=0)
            for idx, value in enumerate(losses, start=1):
                rows.append(
                    {
                        "hidden_dim": hidden_dim,
                        "tokens": int(tokens),
                        "n": idx,
                        "loss": float(value),
                    }
                )
    if not rows:
        raise RuntimeError("No n-gram losses available for any run/checkpoint")
    return pd.DataFrame(rows)


def plot_vs_hidden_dim_by_tokens(
    points: pd.DataFrame,
    value_col: str,
    ylabel: str,
    out_path: str,
    *,
    fit: bool = True,
) -> None:
    """`value_col` vs hidden_dim, one series per token checkpoint.

    Colour encodes training tokens rather than hidden_dim, since hidden_dim is
    the x-axis here. Markers/edges follow the conventions in plot_L_infinity.
    """
    from analysis.plot_bipartite_mi import _format_token_count

    token_values = sorted(points["tokens"].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(token_values), vmax=max(token_values))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    for idx, tokens in enumerate(token_values):
        series = points[points["tokens"] == tokens].sort_values("hidden_dim")
        hd = series["hidden_dim"].to_numpy(dtype=float)
        y = series[value_col].to_numpy(dtype=float)
        color = cmap(norm(tokens))
        ax.scatter(
            hd,
            y,
            marker=markers[idx % len(markers)],
            edgecolors="black",
            color=color,
            alpha=0.8,
        )
        mask = (hd > 0) & (y > 0) & np.isfinite(hd) & np.isfinite(y)
        if fit and int(np.sum(mask)) >= 2:
            power, log_coef = np.polyfit(np.log(hd[mask]), np.log(y[mask]), 1)
            coef = float(np.exp(log_coef))
            fit_x = np.logspace(
                np.log10(float(hd[mask].min())), np.log10(float(hd[mask].max())), 200
            )
            ax.plot(fit_x, coef * np.power(fit_x, power), linestyle=":", color=color)
            print(
                f"fit tokens={_format_token_count(int(tokens))}: "
                f"{value_col}={coef:.4g}*d_h^{power:.4f} "
                f"({int(np.sum(mask))} points)"
            )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02
    )
    colorbar.set_label("training tokens")
    colorbar.set_ticks(token_values)
    colorbar.set_ticklabels([_format_token_count(int(t)) for t in token_values])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_h$")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved to {out_path}")
    _show_image(out_path)

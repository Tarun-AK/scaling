"""Plot bipartite mutual information I(A:B) vs |B| with fixed |A|.

This mirrors analysis/plot_bipartite_mi_fixed_b.py, except the first partition
size |A| is fixed and the second partition size |B| varies.

Usage:
  python analysis/plot_bipartite_mi_fixed_a.py --group <group>
  python analysis/plot_bipartite_mi_fixed_a.py --group <group> --estimator direct v-club
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import wandb
from analysis.plot_bipartite_mi import (
    DEFAULT_MAX_N,
    DEFAULT_MIN_N,
    DEFAULT_N_VALUES,
    FIT_NMAX,
    FIT_NMIN,
    _data_cache_key,
    _data_cache_paths,
    _data_estimator_cache_path,
    _compute_v_club_from_samples,
    _dedupe_estimators,
    _download_checkpoint_artifact,
    _estimator_cache_path_for_sample_cache,
    _filter_runs_by_hidden_dim,
    _find_reusable_complete_cache,
    _load_checkpoint,
    _load_estimator_cache,
    _load_data_chunks,
    _load_data_sample_cache,
    _load_log_q_y_mean_cache,
    _load_sample_cache,
    _load_sample_logps,
    _normalize_params_for_step,
    _plot_bipartite_mi,
    _resolve_group_runs,
    _sample_cache_key,
    _sample_cache_paths,
    _save_data_sample_cache,
    _save_estimator_cache,
    _save_log_q_y_mean_cache,
    _save_sample_cache,
    _sample_sequences,
    _score_logps,
    _select_cached_n_values,
    _validate_args,
)

FIXED_A_SIZE = 800


def _compute_log_q_y_means_fixed_a(
    *,
    samples: np.ndarray,
    apply_fn,
    params: dict,
    n_values: list[int],
    a_size: int,
    batch_size: int,
    bos_token_id: int,
    progress_desc_prefix: str,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for n_b in n_values:
        y_tokens = samples[:, a_size : a_size + n_b]
        y_logps = _score_logps(
            apply_fn=apply_fn,
            params=params,
            tokens=y_tokens,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=f"{progress_desc_prefix} |B|={n_b}",
        )
        out[n_b] = float(np.mean(np.sum(y_logps, axis=1)))
    return out


def _compute_bipartite_mi_from_sampled_q_fixed_a(
    *,
    sample_logps: np.ndarray,
    n_values: list[int],
    log_q_y_means_by_n_b: dict[int, float],
    a_size: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for n_b in n_values:
        if n_b not in log_q_y_means_by_n_b:
            raise RuntimeError(f"Missing cached log q(y) for |B|={n_b}")
        if a_size + n_b > int(sample_logps.shape[1]):
            raise RuntimeError(
                f"|B|={n_b} with |A|={a_size} exceeds sampled seq len"
            )
        log_q_y_given_x = np.mean(np.sum(sample_logps[:, a_size : a_size + n_b], axis=1))
        out[n_b] = float(log_q_y_given_x - log_q_y_means_by_n_b[n_b])
    return out


def _compute_v_club_from_samples_fixed_a(
    *,
    samples: np.ndarray,
    sample_logps: np.ndarray,
    n_values: list[int],
    a_size: int,
    apply_fn,
    params: dict,
    batch_size: int,
    bos_token_id: int,
    rng: np.random.Generator | None = None,
) -> dict[int, float]:
    if rng is None:
        rng = np.random.default_rng(0)

    num_samples = int(sample_logps.shape[0])
    if num_samples < 2:
        raise RuntimeError("Need at least 2 samples for v-club estimator")

    out: dict[int, float] = {}
    for n_b in n_values:
        if a_size + n_b > int(sample_logps.shape[1]):
            continue
        log_q_y_given_x = np.mean(np.sum(sample_logps[:, a_size : a_size + n_b], axis=1))

        tokens_n = samples[:, : a_size + n_b]
        x_tokens = tokens_n[:, :a_size]
        y_tokens = tokens_n[:, a_size : a_size + n_b]
        shuffled_indices = rng.permutation(num_samples)
        x_tokens_shuffled = x_tokens[shuffled_indices]
        cross_tokens = np.concatenate([x_tokens_shuffled, y_tokens], axis=1)
        cross_logps = _score_logps(
            apply_fn=apply_fn,
            params=params,
            tokens=cross_tokens,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=f"Scoring v-club cross |B|={n_b}",
        )
        log_q_y_given_x_shuffled = np.mean(np.sum(cross_logps[:, a_size : a_size + n_b], axis=1))
        out[n_b] = float(log_q_y_given_x - log_q_y_given_x_shuffled)
    return out


def _compute_lstm_sampled_mi_for_run_fixed_a(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    hidden_dim: int,
    n_values: list[int],
    *,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> dict[int, float]:
    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    seq_len_total = int(FIXED_A_SIZE + n_values[-1])
    sample_key = _sample_cache_key(
        seq_len=seq_len_total,
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
            seq_len=seq_len_total,
            target_num_samples=num_samples,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            n_values=n_values,
        )
        if reusable_cache is None:
            raise RuntimeError(
                "No complete cached direct MI found for "
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
                "Cached direct artifacts have no usable |B| values for "
                f"hidden_dim={hidden_dim}. Re-run with --force-resample to regenerate."
            )
        print(
            "Using cached direct MI for hidden_dim="
            f"{hidden_dim} from {os.path.basename(reusable_sample_cache_path)} "
            f"(num_samples={reusable_num_samples}, requested={num_samples})"
        )
        return _compute_bipartite_mi_from_sampled_q_fixed_a(
            sample_logps=reusable_sample_logps,
            n_values=available_n_values,
            log_q_y_means_by_n_b=reusable_log_q_y_means_by_n,
            a_size=FIXED_A_SIZE,
        )

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
        seq_len=int(cfg.get("seq_len", seq_len_total)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )
    state = create_train_state(model, state_cfg, rng)
    state, restored = _load_checkpoint(ckpt_path, state)
    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    sample_params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))
    samples, sample_logps = _sample_sequences(
        model=model,
        params=sample_params,
        seq_len=seq_len_total,
        num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        rng=rng,
        progress_desc=f"Sampling d_h={hidden_dim}",
    )
    _save_sample_cache(sample_cache_path, samples, sample_logps)

    log_q_y_means_by_n = _compute_log_q_y_means_fixed_a(
        samples=samples,
        apply_fn=state.apply_fn,
        params=sample_params,
        n_values=n_values,
        a_size=FIXED_A_SIZE,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        progress_desc_prefix=f"Scoring y d_h={hidden_dim}",
    )
    _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)

    return _compute_bipartite_mi_from_sampled_q_fixed_a(
        sample_logps=sample_logps,
        n_values=n_values,
        log_q_y_means_by_n_b=log_q_y_means_by_n,
        a_size=FIXED_A_SIZE,
    )


def _compute_lstm_direct_mi_for_run_fixed_a_from_data(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    hidden_dim: int,
    n_values: list[int],
    *,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
    data_split: str,
    data_seed: int,
) -> dict[int, float]:
    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    seq_len_required = int(FIXED_A_SIZE + n_values[-1])
    seq_len_data = int(cfg.get("seq_len", seq_len_required))
    if seq_len_required > seq_len_data:
        raise RuntimeError(
            "Data sample source requires FIXED_A_SIZE+max(|B|) <= config.seq_len. "
            f"Got required={seq_len_required}, config.seq_len={seq_len_data}"
        )

    data_key = _data_cache_key(
        split=data_split,
        seq_len=seq_len_required,
        num_samples=num_samples,
        bos_token_id=bos_token_id,
        seed=data_seed,
    )
    sample_cache_path, log_q_y_cache_path = _data_cache_paths(
        cache_dir,
        run.id,
        data_key,
    )

    if (not force_resample) and os.path.exists(sample_cache_path) and os.path.exists(
        log_q_y_cache_path
    ):
        cached = _load_data_sample_cache(sample_cache_path)
        if cached is not None:
            samples, sample_logps = cached
            log_q_y_means_by_n = _load_log_q_y_mean_cache(log_q_y_cache_path)
            available_n_values = [
                int(n)
                for n in n_values
                if int(n) in log_q_y_means_by_n
                and int(FIXED_A_SIZE + int(n)) <= int(sample_logps.shape[1])
            ]
            if available_n_values:
                print(
                    "Using cached data direct MI for hidden_dim="
                    f"{hidden_dim} from {os.path.basename(sample_cache_path)} "
                    f"(num_samples={samples.shape[0]})"
                )
                return _compute_bipartite_mi_from_sampled_q_fixed_a(
                    sample_logps=sample_logps,
                    n_values=available_n_values,
                    log_q_y_means_by_n_b=log_q_y_means_by_n,
                    a_size=FIXED_A_SIZE,
                )

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
        seq_len=int(cfg.get("seq_len", seq_len_required)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )
    state = create_train_state(model, state_cfg, rng)
    state, restored = _load_checkpoint(ckpt_path, state)
    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    sample_params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))

    data_rng = np.random.default_rng(int(data_seed))
    samples_full = _load_data_chunks(
        cfg=cfg,
        split=data_split,
        num_samples=num_samples,
        rng=data_rng,
    )
    samples = np.array(samples_full[:, :seq_len_required], dtype=np.int32)
    sample_logps = _score_logps(
        apply_fn=state.apply_fn,
        params=sample_params,
        tokens=samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        progress_desc=f"Scoring data samples d_h={hidden_dim}",
    )
    _save_data_sample_cache(sample_cache_path, samples, sample_logps)

    log_q_y_means_by_n = _compute_log_q_y_means_fixed_a(
        samples=samples,
        apply_fn=state.apply_fn,
        params=sample_params,
        n_values=n_values,
        a_size=FIXED_A_SIZE,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        progress_desc_prefix=f"Scoring y (data) d_h={hidden_dim}",
    )
    _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)

    return _compute_bipartite_mi_from_sampled_q_fixed_a(
        sample_logps=sample_logps,
        n_values=n_values,
        log_q_y_means_by_n_b=log_q_y_means_by_n,
        a_size=FIXED_A_SIZE,
    )


def _compute_lstm_v_club_for_run_fixed_a_from_data(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    hidden_dim: int,
    n_values: list[int],
    *,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
    data_split: str,
    data_seed: int,
) -> dict[int, float]:
    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    seq_len_required = int(FIXED_A_SIZE + n_values[-1])
    seq_len_data = int(cfg.get("seq_len", seq_len_required))
    if seq_len_required > seq_len_data:
        raise RuntimeError(
            "Data sample source requires FIXED_A_SIZE+max(|B|) <= config.seq_len. "
            f"Got required={seq_len_required}, config.seq_len={seq_len_data}"
        )

    data_key = _data_cache_key(
        split=data_split,
        seq_len=seq_len_required,
        num_samples=num_samples,
        bos_token_id=bos_token_id,
        seed=data_seed,
    )
    sample_cache_path, _ = _data_cache_paths(
        cache_dir,
        run.id,
        data_key,
    )
    vclub_cache_path = _data_estimator_cache_path(
        cache_dir,
        run.id,
        data_key,
        "v-club",
    )

    if not force_resample:
        cached = _load_data_sample_cache(sample_cache_path)
        if cached is not None and os.path.exists(vclub_cache_path):
            _, sample_logps = cached
            cached_vclub = _load_estimator_cache(vclub_cache_path)
            available_n_values = [
                int(n)
                for n in n_values
                if int(n) in cached_vclub
                and int(FIXED_A_SIZE + int(n)) <= int(sample_logps.shape[1])
            ]
            if available_n_values:
                print(
                    "Using cached data v-club for hidden_dim="
                    f"{hidden_dim} from {os.path.basename(vclub_cache_path)}"
                )
                return {int(n): float(cached_vclub[int(n)]) for n in available_n_values}

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
        seq_len=int(cfg.get("seq_len", seq_len_required)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )
    state = create_train_state(model, state_cfg, rng)
    state, restored = _load_checkpoint(ckpt_path, state)
    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    sample_params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))

    cached = None if force_resample else _load_data_sample_cache(sample_cache_path)
    if cached is None:
        data_rng = np.random.default_rng(int(data_seed))
        samples_full = _load_data_chunks(
            cfg=cfg,
            split=data_split,
            num_samples=num_samples,
            rng=data_rng,
        )
        samples = np.array(samples_full[:, :seq_len_required], dtype=np.int32)
        sample_logps = _score_logps(
            apply_fn=state.apply_fn,
            params=sample_params,
            tokens=samples,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=f"Scoring data samples d_h={hidden_dim}",
        )
        _save_data_sample_cache(sample_cache_path, samples, sample_logps)
    else:
        samples, sample_logps = cached

    available_n_values = [
        int(n)
        for n in n_values
        if int(FIXED_A_SIZE + int(n)) <= int(sample_logps.shape[1])
    ]
    if not available_n_values:
        raise RuntimeError(f"No usable |B| values for data v-club hidden_dim={hidden_dim}")

    cached_vclub_values = (
        {} if force_resample else _load_estimator_cache(vclub_cache_path)
    )
    missing_n_values = [n for n in available_n_values if int(n) not in cached_vclub_values]
    if missing_n_values:
        computed_vclub_values = _compute_v_club_from_samples_fixed_a(
            samples=samples,
            sample_logps=sample_logps,
            n_values=missing_n_values,
            a_size=FIXED_A_SIZE,
            apply_fn=state.apply_fn,
            params=sample_params,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            rng=np.random.default_rng(0),
        )
        cached_vclub_values.update(computed_vclub_values)
        _save_estimator_cache(vclub_cache_path, cached_vclub_values)

    out = {
        int(n): float(cached_vclub_values[int(n)])
        for n in available_n_values
        if int(n) in cached_vclub_values
    }
    if not out:
        raise RuntimeError(
            f"No cached/computed data v-club values for hidden_dim={hidden_dim}"
        )
    return out


def _compute_lstm_v_club_for_run_fixed_a(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    hidden_dim: int,
    n_values: list[int],
    *,
    num_samples: int,
    batch_size: int,
    cache_dir: str,
    force_resample: bool,
) -> dict[int, float]:
    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    seq_len_total = int(FIXED_A_SIZE + n_values[-1])

    reusable_cache = _find_reusable_complete_cache(
        cache_dir=cache_dir,
        run_id=run.id,
        seq_len=seq_len_total,
        target_num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        n_values=n_values,
    )
    if reusable_cache is None and force_resample:
        _compute_lstm_sampled_mi_for_run_fixed_a(
            run,
            api,
            hidden_dim,
            n_values,
            num_samples=num_samples,
            batch_size=batch_size,
            cache_dir=cache_dir,
            force_resample=True,
        )
        reusable_cache = _find_reusable_complete_cache(
            cache_dir=cache_dir,
            run_id=run.id,
            seq_len=seq_len_total,
            target_num_samples=num_samples,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            n_values=n_values,
        )
    if reusable_cache is None:
        raise RuntimeError(
            "No complete cached direct artifacts found for "
            f"hidden_dim={hidden_dim}. Re-run with --force-resample."
        )

    sample_logps = reusable_cache[0]
    sample_cache_path = reusable_cache[3]
    available_n_values = [
        n
        for n in n_values
        if int(FIXED_A_SIZE + n) <= int(sample_logps.shape[1])
    ]
    if not available_n_values:
        raise RuntimeError(f"No usable |B| values for v-club hidden_dim={hidden_dim}")

    cached_samples = _load_sample_cache(sample_cache_path)
    if cached_samples is None:
        raise RuntimeError(f"Failed to load cached samples from {sample_cache_path}")
    samples, cached_sample_logps = cached_samples

    vclub_cache_path = _estimator_cache_path_for_sample_cache(sample_cache_path, "v-club")
    cached_vclub_values = _load_estimator_cache(vclub_cache_path)
    missing_n_values = [n for n in available_n_values if int(n) not in cached_vclub_values]

    if missing_n_values:
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
            seq_len=int(cfg.get("seq_len", seq_len_total)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
        )
        state = create_train_state(model, state_cfg, rng)
        state, restored = _load_checkpoint(ckpt_path, state)
        ckpt_run_id = restored.get("wandb_run_id")
        if ckpt_run_id != run.id:
            raise RuntimeError(
                "Checkpoint/run mismatch: "
                f"ckpt_run_id={ckpt_run_id}, run.id={run.id}"
            )
        sample_params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))

        computed_vclub_values = _compute_v_club_from_samples_fixed_a(
            samples=samples,
            sample_logps=cached_sample_logps,
            n_values=missing_n_values,
            a_size=FIXED_A_SIZE,
            apply_fn=state.apply_fn,
            params=sample_params,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            rng=np.random.default_rng(0),
        )
        cached_vclub_values.update(computed_vclub_values)
        _save_estimator_cache(vclub_cache_path, cached_vclub_values)

    out = {
        int(n): float(cached_vclub_values[int(n)])
        for n in available_n_values
        if int(n) in cached_vclub_values
    }
    if not out:
        raise RuntimeError(f"No cached/computed v-club values for hidden_dim={hidden_dim}")
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["direct", "v-club"],
        nargs="+",
        default=["direct"],
        help="One or more estimators: direct v-club",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        nargs="+",
        default=None,
        help="Optional hidden_dim filter(s)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help="Maximum |B| to include",
    )
    parser.add_argument(
        "--fit-nmin",
        type=int,
        default=FIT_NMIN,
        help="Min |B| to include in power-law fits (|B| >= fit_nmin)",
    )
    parser.add_argument(
        "--fit-nmax",
        type=int,
        default=FIT_NMAX,
        help=(
            "Max |B| to include in power-law fits (|B| < fit_nmax). "
            "Use <=0 for no upper bound"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of sampled sequences",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sampling and scoring",
    )
    parser.add_argument(
        "--sample-source",
        type=str,
        choices=["model", "data"],
        default="model",
        help="Use model samples or dataset chunks",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["validation", "train", "test"],
        default="validation",
        help="Which cached dataset split to draw chunks from when --sample-source=data",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="RNG seed for selecting data chunks when --sample-source=data",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_fixed_a_cache",
        help="Directory for MI cache",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Force regeneration of direct caches instead of cache-only mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)
    estimators = _dedupe_estimators(args.estimator)

    n_values = [n for n in DEFAULT_N_VALUES if n <= int(args.max_n)]
    if not n_values:
        raise RuntimeError("No valid |B| values to evaluate")
    if min(n_values) < DEFAULT_MIN_N:
        raise RuntimeError(f"--max-n must keep |B| >= {DEFAULT_MIN_N}")

    api = wandb.Api()
    runs = _resolve_group_runs(api, args.group)
    runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, args.group)

    all_values: dict[str, dict[int, dict[int, float]]] = {
        estimator: {} for estimator in estimators
    }
    for run in tqdm(runs, desc="Runs", unit="run"):
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])

        if "direct" in estimators:
            if args.sample_source == "data":
                all_values["direct"][
                    hidden_dim
                ] = _compute_lstm_direct_mi_for_run_fixed_a_from_data(
                    run,
                    api,
                    hidden_dim,
                    n_values,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                    data_split=args.data_split,
                    data_seed=args.data_seed,
                )
            else:
                all_values["direct"][
                    hidden_dim
                ] = _compute_lstm_sampled_mi_for_run_fixed_a(
                    run,
                    api,
                    hidden_dim,
                    n_values,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                )

        if "v-club" in estimators:
            if args.sample_source == "data":
                all_values["v-club"][
                    hidden_dim
                ] = _compute_lstm_v_club_for_run_fixed_a_from_data(
                    run,
                    api,
                    hidden_dim,
                    n_values,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                    data_split=args.data_split,
                    data_seed=args.data_seed,
                )
            else:
                all_values["v-club"][
                    hidden_dim
                ] = _compute_lstm_v_club_for_run_fixed_a(
                    run,
                    api,
                    hidden_dim,
                    n_values,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    force_resample=args.force_resample,
                )

    out_path = (
        args.output
        if args.output is not None
        else f"results/bipartite_mi_fixed_a{FIXED_A_SIZE}_{args.group}.png"
    )
    _plot_bipartite_mi(
        all_values,
        estimators,
        out_path,
        title=(
            "Bipartite MI fixed-|A| "
            f"(|A|={FIXED_A_SIZE}, {', '.join(estimators)}, group={args.group})"
        ),
        fit_nmin=args.fit_nmin,
        fit_nmax=args.fit_nmax,
    )


if __name__ == "__main__":
    main()

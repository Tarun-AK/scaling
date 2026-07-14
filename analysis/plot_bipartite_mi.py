"""Plot bipartite mutual information I(A:B) vs N.

Usage:
  python analysis/plot_bipartite_mi.py --group <group>
  python analysis/plot_bipartite_mi.py --group <group> --estimator direct v-club
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
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm

import wandb
from analysis.plot_group_labels import distinct_group_labels

WANDB_PROJECT = "tarunadvaith-/scaling"
DEFAULT_N_VALUES = [
    # 6,
    # 8,
    # 10,
    # 12,
    14,
    # 16,
    # 20,
    # 22,
    26,
    # 32,
    # 38,
    # 44,
    52,
    # 62,
    # 74,
    # 88,
    104,
    # 122,
    # 146,
    172,
    # 206,
    244,
    # 288,
    # 342,
    406,
    # 482,
    572,
    # 680,
    806,
    # 958,
    1136,
    # 1348,
    # 1600,
    # 1800,
    2048,
    3000,
    4096,
    5000,
    6000,
    7000,
    8192,
]
DEFAULT_MIN_N = min(DEFAULT_N_VALUES)
DEFAULT_MAX_N = 2048
DEFAULT_NUM_N_VALUES = 40
FIT_NMAX = 128
FIT_NMIN = 1
plt.style.use("~/plotStyle.mplstyle")


def _data_cache_key(
    *,
    split: str,
    seq_len: int,
    num_samples: int | None,
    bos_token_id: int,
    seed: int | None,
) -> str:
    if num_samples is None:
        return f"data_{split}_seq{int(seq_len)}_numall_bos{int(bos_token_id)}"
    return (
        f"data_{split}_seq{int(seq_len)}_num{int(num_samples)}_"
        f"bos{int(bos_token_id)}_seed{int(seed or 0)}"
    )


def _data_cache_paths(
    cache_dir: str,
    run_id: str,
    data_key: str,
) -> tuple[str, str]:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    os.makedirs(run_cache_dir, exist_ok=True)
    sample_cache_path = os.path.join(run_cache_dir, f"data_samples_{data_key}.npz")
    log_q_y_cache_path = os.path.join(run_cache_dir, f"data_log_q_y_{data_key}.npz")
    return sample_cache_path, log_q_y_cache_path


def _data_estimator_cache_path(
    cache_dir: str,
    run_id: str,
    data_key: str,
    estimator: str,
) -> str:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    os.makedirs(run_cache_dir, exist_ok=True)
    safe_estimator = estimator.replace("-", "")
    return os.path.join(run_cache_dir, f"data_{safe_estimator}_{data_key}.npz")


def _load_data_sample_cache(
    sample_cache_path: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(sample_cache_path):
        return None
    with np.load(sample_cache_path) as data:
        if "samples" not in data or "sample_logps" not in data:
            return None
        samples = np.array(data["samples"])
        sample_logps = np.array(data["sample_logps"])
    if samples.shape != sample_logps.shape:
        return None
    return samples, sample_logps


def _save_data_sample_cache(
    sample_cache_path: str,
    samples: np.ndarray,
    sample_logps: np.ndarray,
) -> None:
    np.savez_compressed(
        sample_cache_path,
        samples=samples,
        sample_logps=sample_logps,
    )


def _load_data_chunks(
    *,
    cfg: dict,
    split: str,
    num_samples: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    from data.dataset import load_splits_as_arrays

    dataset_config = cfg.get("dataset_config")
    dataset_path = cfg.get("dataset_path")
    train_np, val_np, test_np = load_splits_as_arrays(
        dataset_name=str(cfg["dataset_name"]),
        dataset_config=dataset_config,
        seq_len=int(cfg["seq_len"]),
        vocab_size=int(cfg["vocab_size"]),
        cache_dir=str(cfg.get("cache_dir", "data/cache")),
        require_cache=bool(cfg.get("require_cached_data", True)),
        tokenize_batch_size=int(cfg.get("tokenize_batch_size", 32)),
        tokenizer_path=str(cfg.get("tokenizer_path", "data/tokenizer/tokenizer.json")),
        dataset_path=str(dataset_path) if dataset_path is not None else None,
    )

    if split == "train":
        data = train_np
    elif split == "test":
        data = test_np
    elif split in {"validation+test", "test+validation"}:
        data = np.concatenate([val_np, test_np], axis=0)
    else:
        data = val_np

    num_rows = int(data.shape[0])
    if num_rows < 1:
        raise RuntimeError(f"No cached rows available for split='{split}'")

    if num_samples is None:
        return np.asarray(data, dtype=np.int32)

    take = int(num_samples)
    if take <= num_rows:
        if rng is None:
            raise RuntimeError("rng must be provided when num_samples is set")
        idx = rng.choice(num_rows, size=take, replace=False)
    else:
        if rng is None:
            raise RuntimeError("rng must be provided when num_samples is set")
        idx = rng.choice(num_rows, size=take, replace=True)
    return np.array(data[idx], dtype=np.int32)


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
    for run in runs:
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


def _wandb_entity_project() -> tuple[str, str]:
    entity, project = WANDB_PROJECT.split("/")
    return entity, project


def _download_checkpoint_artifact(
    run_id: str,
    api: wandb.Api,
    cache_dir: str,
) -> str:
    import re

    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cached_ckpt = os.path.join(cache_dir, run_id, "ckpt")
    if os.path.exists(cached_ckpt):
        return cached_ckpt

    entity, project = _wandb_entity_project()
    artifact_name = f"{entity}/{project}/checkpoint-{run_id}:latest"
    try:
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download(root=os.path.join(cache_dir, run_id))
        return os.path.join(artifact_dir, "ckpt")
    except Exception:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        pattern = re.compile(rf"^checkpoint-{re.escape(run_id)}-epoch-(\d{{4}}):v\d+$")
        best_artifact_name = None
        best_epoch = -1
        for logged_artifact in run.logged_artifacts():
            match = pattern.match(logged_artifact.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_artifact_name = logged_artifact.name

        if best_artifact_name is None:
            raise

        print(
            "Falling back to latest epoch checkpoint artifact for "
            f"run_id={run_id}: {best_artifact_name}"
        )
        epoch_artifact = api.artifact(f"{WANDB_PROJECT}/{best_artifact_name}")
        artifact_dir = epoch_artifact.download(root=os.path.join(cache_dir, run_id))
        return os.path.join(artifact_dir, "ckpt")


def _load_checkpoint(ckpt_path: str, state) -> tuple[Any, dict]:
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return state.replace(params=restored["params"]), restored


def _normalize_params_for_step(params: dict, num_layers: int) -> dict:
    if any(k.startswith("rnn_") for k in params):
        return params

    lstm_keys = [k for k in params if k.startswith("LSTMCell_")]
    if not lstm_keys:
        return params

    normalized = dict(params)
    for layer_idx in range(num_layers):
        cell_key = f"LSTMCell_{layer_idx}"
        if cell_key not in params:
            raise RuntimeError(
                f"Missing {cell_key} in params for num_layers={num_layers}"
            )
        normalized.setdefault(f"rnn_{layer_idx}", {"cell": params[cell_key]})
    return normalized


def _build_language_model_from_config(cfg: dict[str, Any]):
    model_type = str(cfg.get("model_type", "lstm")).lower()
    if model_type == "lstm":
        from models.lstm import LSTMLanguageModel

        return LSTMLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    if model_type == "mamba":
        from models.mamba import MambaLanguageModel

        return MambaLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
            state_size=int(cfg.get("mamba_state_size", cfg.get("mamba_d_state", 16))),
            head_dim=int(cfg.get("mamba_head_dim", 16)),
            chunk_size=int(cfg.get("mamba_chunk_size", 256)),
            expand=int(cfg.get("mamba_expand", 2)),
            conv_kernel=int(cfg.get("mamba_conv_kernel", 4)),
        )
    if model_type == "vanilla_rnn":
        from models.vanilla_rnn import VanillaRNNLanguageModel

        return VanillaRNNLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
        )
    if model_type == "transformer":
        from models.transformer import TransformerLanguageModel

        num_heads = int(cfg.get("transformer_num_heads", 8))
        return TransformerLanguageModel(
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
            num_heads=num_heads,
            num_kv_heads=int(cfg.get("transformer_num_kv_heads", num_heads)),
            ffn_mult=float(cfg.get("transformer_ffn_mult", 8.0 / 3.0)),
            rope_theta=float(cfg.get("transformer_rope_theta", 1_000_000.0)),
            layer_norm_epsilon=float(cfg.get("transformer_layer_norm_epsilon", 1e-6)),
            tie_word_embeddings=bool(cfg.get("transformer_tie_word_embeddings", True)),
        )
    raise RuntimeError(
        "Unsupported model_type in run config. Expected one of "
        f"['lstm', 'mamba', 'vanilla_rnn', 'transformer'], got '{model_type}'"
    )


def _sample_sequences(
    model,
    params: dict,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
    rng,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    def _sample_batch(
        batch_rng,
        init_carry: tuple,
        seq_len: int,
        bos_token_id: int,
    ):
        def scan_step(state, _):
            next_carry, logits = model.apply(
                {"params": params},
                state["carry"],
                state["token"],
                method=model.step,
            )
            rng, step_rng = jax.random.split(state["rng"])
            next_token = jax.random.categorical(step_rng, logits)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_logp = jnp.take_along_axis(
                log_probs,
                next_token[:, None],
                axis=-1,
            ).squeeze(-1)
            new_state = {"carry": next_carry, "token": next_token, "rng": rng}
            return new_state, (next_token, token_logp)

        batch_size_local = int(jax.tree_util.tree_leaves(init_carry)[0].shape[0])
        init_token = jnp.full((batch_size_local,), bos_token_id)
        carry = {"carry": init_carry, "token": init_token, "rng": batch_rng}
        _, (tokens, logps) = jax.lax.scan(scan_step, carry, jnp.arange(seq_len))
        return jnp.transpose(tokens, (1, 0)), jnp.transpose(logps, (1, 0))

    sample_batch_jit = jax.jit(
        _sample_batch,
        static_argnames=("seq_len", "bos_token_id"),
    )

    all_samples = []
    all_logps = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(
        range(num_batches),
        desc=progress_desc,
        unit="batch",
        leave=False,
    ):
        rng, batch_rng = jax.random.split(rng)
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        init_carry = model.init_carry(current_batch_size)
        tokens, logps = sample_batch_jit(batch_rng, init_carry, seq_len, bos_token_id)
        all_samples.append(np.array(tokens))
        all_logps.append(np.array(logps))

    samples = np.concatenate(all_samples, axis=0)[:num_samples]
    sample_logps = np.concatenate(all_logps, axis=0)[:num_samples]
    return samples, sample_logps


def _prepend_bos(tokens, bos_token_id: int):
    import jax.numpy as jnp

    bos = jnp.full((tokens.shape[0], 1), bos_token_id, dtype=tokens.dtype)
    return jnp.concatenate([bos, tokens[:, :-1]], axis=1)


def _score_logps(
    apply_fn,
    params: dict,
    tokens,
    batch_size: int,
    bos_token_id: int,
    progress_desc: str,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    tokens_device = jax.device_put(tokens)
    num_tokens = int(tokens_device.shape[0])
    if num_tokens == 0:
        seq_len = int(tokens_device.shape[1])
        return np.empty((0, seq_len), dtype=np.float32)

    fixed_batch_size = min(batch_size, num_tokens)

    @jax.jit
    def _score_batch(batch_tokens: jax.Array, batch_params: dict) -> jax.Array:
        inputs = _prepend_bos(batch_tokens, bos_token_id)
        logits = apply_fn({"params": batch_params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(
            log_probs,
            batch_tokens[:, :, None],
            axis=-1,
        ).squeeze(-1)

    all_logps = []
    num_batches = (num_tokens + batch_size - 1) // batch_size
    for batch_idx in tqdm(
        range(num_batches),
        desc=progress_desc,
        unit="batch",
        leave=False,
    ):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_tokens)
        current_size = end - start
        batch = tokens_device[start:end]
        if current_size < fixed_batch_size:
            pad_rows = fixed_batch_size - current_size
            pad = jnp.repeat(batch[:1], pad_rows, axis=0)
            batch = jnp.concatenate([batch, pad], axis=0)

        token_logp = _score_batch(batch, params)
        all_logps.append(np.array(token_logp[:current_size]))

    return np.concatenate(all_logps, axis=0)[:num_tokens]


def _sample_cache_key(
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
) -> str:
    return (
        f"seq{int(seq_len)}_num{int(num_samples)}_"
        f"samplebs{int(batch_size)}_bos{int(bos_token_id)}"
    )


def _sample_cache_paths(
    cache_dir: str,
    run_id: str,
    sample_key: str,
) -> tuple[str, str]:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    os.makedirs(run_cache_dir, exist_ok=True)
    sample_cache_path = os.path.join(run_cache_dir, f"samples_{sample_key}.npz")
    log_q_y_cache_path = os.path.join(run_cache_dir, f"log_q_y_{sample_key}.npz")
    return sample_cache_path, log_q_y_cache_path


def _sample_key_from_sample_cache_path(sample_cache_path: str) -> str | None:
    filename = os.path.basename(sample_cache_path)
    if not filename.startswith("samples_") or not filename.endswith(".npz"):
        return None
    return filename[len("samples_") : -len(".npz")]


def _log_q_y_cache_path_for_sample_cache(sample_cache_path: str) -> str:
    sample_key = _sample_key_from_sample_cache_path(sample_cache_path)
    if sample_key is None:
        raise RuntimeError(f"Invalid sample cache filename: {sample_cache_path}")
    return os.path.join(
        os.path.dirname(sample_cache_path),
        f"log_q_y_{sample_key}.npz",
    )


def _estimator_cache_path_for_sample_cache(
    sample_cache_path: str,
    estimator: str,
) -> str:
    sample_key = _sample_key_from_sample_cache_path(sample_cache_path)
    if sample_key is None:
        raise RuntimeError(f"Invalid sample cache filename: {sample_cache_path}")
    safe_estimator = estimator.replace("-", "")
    return os.path.join(
        os.path.dirname(sample_cache_path),
        f"{safe_estimator}_{sample_key}.npz",
    )


def _load_sample_cache(sample_cache_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(sample_cache_path):
        return None
    with np.load(sample_cache_path) as data:
        if "samples" not in data or "sample_logps" not in data:
            return None
        samples = np.array(data["samples"])
        sample_logps = np.array(data["sample_logps"])
    if samples.shape != sample_logps.shape:
        return None
    return samples, sample_logps


def _load_sample_logps(sample_cache_path: str) -> np.ndarray | None:
    if not os.path.exists(sample_cache_path):
        return None
    with np.load(sample_cache_path) as data:
        if "sample_logps" not in data:
            return None
        sample_logps = np.array(data["sample_logps"])
    if sample_logps.ndim != 2:
        return None
    return sample_logps


def _save_sample_cache(
    sample_cache_path: str,
    samples: np.ndarray,
    sample_logps: np.ndarray,
) -> None:
    np.savez_compressed(
        sample_cache_path,
        samples=samples,
        sample_logps=sample_logps,
    )


def _load_log_q_y_mean_cache(log_q_y_cache_path: str) -> dict[int, float]:
    if not os.path.exists(log_q_y_cache_path):
        return {}
    with np.load(log_q_y_cache_path) as data:
        if "n_values" not in data or "log_q_y_means" not in data:
            return {}
        n_values = np.array(data["n_values"])
        log_q_y_means = np.array(data["log_q_y_means"])
    if len(n_values) != len(log_q_y_means):
        return {}
    out: dict[int, float] = {}
    for n, value in zip(n_values, log_q_y_means):
        n_int = int(n)
        value_float = float(value)
        if np.isfinite(value_float):
            out[n_int] = value_float
    return out


def _save_log_q_y_mean_cache(
    log_q_y_cache_path: str,
    log_q_y_means_by_n: dict[int, float],
) -> None:
    n_values = np.array(sorted(log_q_y_means_by_n.keys()), dtype=np.int32)
    log_q_y_means = np.array(
        [log_q_y_means_by_n[int(n)] for n in n_values],
        dtype=np.float32,
    )
    np.savez_compressed(
        log_q_y_cache_path,
        n_values=n_values,
        log_q_y_means=log_q_y_means,
    )


def _load_estimator_cache(estimator_cache_path: str) -> dict[int, float]:
    if not os.path.exists(estimator_cache_path):
        return {}
    with np.load(estimator_cache_path) as data:
        if "n_values" not in data or "values" not in data:
            return {}
        n_values = np.array(data["n_values"])
        values = np.array(data["values"])
    if len(n_values) != len(values):
        return {}
    out: dict[int, float] = {}
    for n, value in zip(n_values, values):
        n_int = int(n)
        value_float = float(value)
        if np.isfinite(value_float):
            out[n_int] = value_float
    return out


def _save_estimator_cache(
    estimator_cache_path: str,
    values_by_n: dict[int, float],
) -> None:
    n_values = np.array(sorted(values_by_n.keys()), dtype=np.int32)
    values = np.array([values_by_n[int(n)] for n in n_values], dtype=np.float32)
    np.savez_compressed(
        estimator_cache_path,
        n_values=n_values,
        values=values,
    )


def _select_cached_n_values(
    requested_n_values: list[int],
    log_q_y_means_by_n: dict[int, float],
    sample_logps: np.ndarray,
) -> list[int]:
    if not requested_n_values:
        return []
    max_requested_n = int(max(requested_n_values))
    max_allowed_n = min(max_requested_n, int(sample_logps.shape[1]))
    cached_n_values = sorted(
        int(n)
        for n in log_q_y_means_by_n
        if DEFAULT_MIN_N <= int(n) <= max_allowed_n and int(n) % 2 == 0
    )
    if cached_n_values:
        return cached_n_values
    return sorted(
        n
        for n in requested_n_values
        if n in log_q_y_means_by_n and n <= int(sample_logps.shape[1])
    )


def _compatible_sample_cache_entries(
    cache_dir: str,
    run_id: str,
    seq_len: int,
    batch_size: int | None,
    bos_token_id: int,
) -> list[tuple[int, str, str]]:
    run_cache_dir = os.path.join(os.path.abspath(cache_dir), run_id)
    if not os.path.isdir(run_cache_dir):
        return []

    pattern = re.compile(
        r"^samples_seq(?P<seq>\d+)_num(?P<num>\d+)_"
        rf"samplebs(?P<samplebs>\d+)_bos{int(bos_token_id)}\.npz$"
    )
    entries: list[tuple[int, str, str]] = []
    for filename in os.listdir(run_cache_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        seq = int(match.group("seq"))
        if seq < int(seq_len):
            continue
        samplebs = int(match.group("samplebs"))
        if batch_size is not None and samplebs != int(batch_size):
            continue
        num_samples = int(match.group("num"))
        if num_samples < 1:
            continue
        sample_cache_path = os.path.join(run_cache_dir, filename)
        log_q_y_cache_path = _log_q_y_cache_path_for_sample_cache(sample_cache_path)
        entries.append((num_samples, sample_cache_path, log_q_y_cache_path))
    entries.sort(key=lambda item: item[0])
    return entries


def _find_reusable_complete_cache(
    cache_dir: str,
    run_id: str,
    seq_len: int,
    target_num_samples: int,
    batch_size: int,
    bos_token_id: int,
    n_values: list[int],
) -> tuple[np.ndarray, dict[int, float], int, str, str] | None:
    entries = _compatible_sample_cache_entries(
        cache_dir=cache_dir,
        run_id=run_id,
        seq_len=seq_len,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
    )
    if not entries:
        entries = _compatible_sample_cache_entries(
            cache_dir=cache_dir,
            run_id=run_id,
            seq_len=seq_len,
            batch_size=None,
            bos_token_id=bos_token_id,
        )

    best_ge_target: tuple[np.ndarray, dict[int, float], int, str, str] | None = None
    best_any: tuple[np.ndarray, dict[int, float], int, str, str] | None = None
    for _, sample_cache_path, log_q_y_cache_path in entries:
        log_q_y_means_by_n = _load_log_q_y_mean_cache(log_q_y_cache_path)
        sample_logps = _load_sample_logps(sample_cache_path)
        if sample_logps is None:
            continue
        actual_num_samples = int(sample_logps.shape[0])
        if actual_num_samples < 1:
            continue
        available_n_values = _select_cached_n_values(
            n_values,
            log_q_y_means_by_n,
            sample_logps,
        )
        if not available_n_values:
            continue
        candidate = (
            sample_logps,
            log_q_y_means_by_n,
            actual_num_samples,
            sample_cache_path,
            log_q_y_cache_path,
        )
        if best_any is None or actual_num_samples > best_any[2]:
            best_any = candidate
        if actual_num_samples >= int(target_num_samples):
            if best_ge_target is None or actual_num_samples > best_ge_target[2]:
                best_ge_target = candidate

    if best_ge_target is not None:
        return best_ge_target
    return best_any


def _compute_log_q_y_means(
    samples,
    apply_fn,
    params: dict,
    n_values: list[int],
    batch_size: int,
    bos_token_id: int,
    progress_desc_prefix: str,
    *,
    sanity_check: bool = False,
    sanity_tol: float = 1e-5,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for n in n_values:
        half = n // 2
        y_tokens = samples[:, half:n]
        y_logps = _score_logps(
            apply_fn=apply_fn,
            params=params,
            tokens=y_tokens,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=f"{progress_desc_prefix} N={n}",
        )
        out[n] = float(np.mean(np.sum(y_logps, axis=1)))
        if sanity_check:
            # L_y,t = -E[log q(y_t | y_{<t})] computed on the same scored suffixes.
            l_y_by_pos = -np.mean(y_logps, axis=0)
            expected = -float(np.sum(l_y_by_pos))
            err = abs(out[n] - expected)
            if err > float(sanity_tol):
                print(
                    "Sanity check warning for log q(y) at N="
                    f"{n}: mean(sum logp)={out[n]:.8g}, -sum(L_y)={expected:.8g}, err={err:.3g}"
                )
    return out


def _sanity_check_direct_data_terms(
    *,
    sample_logps: np.ndarray,
    log_q_y_means_by_n: dict[int, float],
    n_values: list[int],
    tol: float = 1e-5,
) -> None:
    sample_logps_cumsum = np.cumsum(sample_logps, axis=1)
    l_by_pos = -np.mean(sample_logps, axis=0)  # positions 1..T
    max_err_cond = 0.0
    max_err_marg = 0.0
    worst_cond: tuple[int, float, float] | None = None
    worst_marg: tuple[int, float, float] | None = None

    for n in n_values:
        if n not in log_q_y_means_by_n:
            continue
        half = n // 2
        cond_code = float(
            np.mean(sample_logps_cumsum[:, n - 1] - sample_logps_cumsum[:, half - 1])
        )
        cond_l = -float(np.sum(l_by_pos[half:n]))
        err_cond = abs(cond_code - cond_l)
        if err_cond > max_err_cond:
            max_err_cond = err_cond
            worst_cond = (n, cond_code, cond_l)

        marg_code = float(log_q_y_means_by_n[n])
        # Stationarity-based substitution target: E[log q(B)] ?= E[log q(A)]
        # and E[log q(A)] = -sum_{t=1..half} L_t, where L_t is computed from
        # the same scored sample_logps.
        marg_l_stationary = -float(np.sum(l_by_pos[:half]))
        err_marg = abs(marg_code - marg_l_stationary)
        if err_marg > max_err_marg:
            max_err_marg = err_marg
            worst_marg = (n, marg_code, marg_l_stationary)

    if worst_cond is None:
        print("Sanity check: no usable N values for direct/data conditional term")
        return

    if max_err_cond <= float(tol):
        print(
            "Sanity check OK (direct/data): "
            f"max |E[log q(B|A)] - (-sum L_n)| = {max_err_cond:.3g}"
        )
    else:
        n, cond_code, cond_l = worst_cond
        print(
            "Sanity check FAILED (direct/data): "
            f"max err={max_err_cond:.3g} at N={n}. "
            f"E[log q(B|A)]={cond_code:.6g}, -sum L_n={cond_l:.6g}"
        )

    if worst_marg is None:
        print("Sanity check: no usable N values for direct/data marginal term")
        return

    if max_err_marg <= float(tol):
        print(
            "Sanity check OK (direct/data, stationarity): "
            f"max |E[log q(B)] - (-sum_{{1..N/2}} L_n)| = {max_err_marg:.3g}"
        )
    else:
        n, marg_code, marg_l = worst_marg
        print(
            "Sanity check WARNING (direct/data, stationarity): "
            f"max err={max_err_marg:.3g} at N={n}. "
            f"E[log q(B)]={marg_code:.6g}, -sum_{{1..N/2}} L_n={marg_l:.6g}"
        )


def _compute_bipartite_mi_from_sampled_q(
    sample_logps: np.ndarray,
    n_values: list[int],
    log_q_y_means_by_n: dict[int, float],
) -> dict[int, float]:
    sample_logps_cumsum = np.cumsum(sample_logps, axis=1)
    out: dict[int, float] = {}
    for n in n_values:
        if n not in log_q_y_means_by_n:
            raise RuntimeError(f"Missing cached log q(y) for N={n}")
        half = n // 2
        log_q_y_given_x = np.mean(
            sample_logps_cumsum[:, n - 1] - sample_logps_cumsum[:, half - 1]
        )
        out[n] = float(log_q_y_given_x - log_q_y_means_by_n[n])
    return out


def _extract_ngram_index(name: str) -> int | None:
    if "/ngram_" in name or "/n_gram_" in name:
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


def _compute_ngram_mi_for_run(
    run: wandb.apis.public.Run,
    n_values: list[int],
) -> dict[int, float]:
    losses_by_n = _extract_combined_ngram_losses(run)
    if not losses_by_n:
        raise RuntimeError(
            "No combined n-gram losses found for run "
            f"{run.id}; cannot compute n-gram MI estimator"
        )

    target_even_ns = sorted(
        {int(n) for n in n_values if int(n) >= 2 and int(n) % 2 == 0}
    )
    even_ns = [n for n in target_even_ns if n in losses_by_n]
    if not even_ns:
        raise RuntimeError(
            "No requested default even N values found in combined n-gram losses for "
            f"run {run.id}; cannot compute n-gram MI estimator"
        )

    max_n = max(even_ns)
    losses = np.full(max_n + 1, np.nan, dtype=float)
    for n, value in losses_by_n.items():
        if 1 <= n <= max_n:
            losses[n] = float(value)

    finite_mask = np.isfinite(losses[1:])
    cumulative = np.cumsum(np.where(finite_mask, losses[1:], 0.0))
    prefix_complete = np.cumprod(finite_mask.astype(np.int8)).astype(bool)

    out: dict[int, float] = {}
    for n in even_ns:
        if not prefix_complete[n - 1]:
            continue
        half = n // 2
        first_sum = float(cumulative[half - 1])
        second_sum = float(cumulative[n - 1] - cumulative[half - 1])
        out[n] = -second_sum + first_sum

    if not out:
        raise RuntimeError(
            "No usable requested default even N values with complete 1..N prefix "
            f"in combined n-gram losses for run {run.id}"
        )
    return out


def _compute_v_club_from_samples(
    *,
    samples: np.ndarray,
    sample_logps: np.ndarray,
    n_values: list[int],
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

    sample_logps_cumsum = np.cumsum(sample_logps, axis=1)
    out: dict[int, float] = {}
    for n in n_values:
        half = n // 2
        if half < 1:
            continue

        log_q_y_given_x = np.mean(
            sample_logps_cumsum[:, n - 1] - sample_logps_cumsum[:, half - 1]
        )

        tokens_n = samples[:, :n]
        x_tokens = tokens_n[:, :half]
        y_tokens = tokens_n[:, half:n]
        shuffled_indices = rng.permutation(num_samples)
        x_tokens_shuffled = x_tokens[shuffled_indices]
        cross_tokens = np.concatenate([x_tokens_shuffled, y_tokens], axis=1)
        cross_logps = _score_logps(
            apply_fn=apply_fn,
            params=params,
            tokens=cross_tokens,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            progress_desc=f"Scoring v-club cross N={n}",
        )
        log_q_y_given_x_shuffled = np.mean(np.sum(cross_logps[:, half:n], axis=1))
        out[n] = float(log_q_y_given_x - log_q_y_given_x_shuffled)
    return out


def _filter_runs_by_hidden_dim(
    runs: list[wandb.apis.public.Run],
    hidden_dim_filter: list[int] | None,
    group: str,
) -> list[wandb.apis.public.Run]:
    if hidden_dim_filter is None:
        return runs
    hidden_dims = set(hidden_dim_filter)
    filtered_runs = [
        run
        for run in runs
        if int((run.config or {}).get("hidden_dim", -1)) in hidden_dims
    ]
    if not filtered_runs:
        raise RuntimeError(
            f"No finished runs found for group='{group}' "
            f"hidden_dim in {sorted(hidden_dims)}"
        )
    return filtered_runs


def _compute_lstm_sampled_mi_for_run(
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
            print("Skipping hidden_dim=" f"{hidden_dim}: no cached direct MI available")
            return {}
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
            print(
                "Skipping hidden_dim="
                f"{hidden_dim}: cached direct MI has no usable N values"
            )
            return {}
        if len(available_n_values) < len(n_values):
            print(
                "Using cached direct MI subset for hidden_dim="
                f"{hidden_dim}: {len(available_n_values)}/{len(n_values)} "
                "N values available"
            )
        print(
            "Using cached direct MI for hidden_dim="
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
        f"{hidden_dim}; regenerating direct cache"
    )
    if os.path.exists(sample_cache_path):
        os.remove(sample_cache_path)
    if os.path.exists(log_q_y_cache_path):
        os.remove(log_q_y_cache_path)

    import jax

    from training.trainer import create_train_state

    ckpt_path = _download_checkpoint_artifact(run.id, api, cache_dir)
    model = _build_language_model_from_config(cfg)
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


def _compute_lstm_direct_mi_for_run_from_data(
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
    sanity_check: bool = False,
) -> dict[int, float]:
    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    seq_len_required = int(n_values[-1])
    seq_len_data = int(cfg.get("seq_len", seq_len_required))
    if seq_len_required > seq_len_data:
        raise RuntimeError(
            "Data sample source requires N <= config.seq_len. "
            f"Got max_n={seq_len_required}, config.seq_len={seq_len_data}"
        )

    data_key = _data_cache_key(
        split=data_split,
        seq_len=seq_len_required,
        num_samples=None,
        bos_token_id=bos_token_id,
        seed=None,
    )
    sample_cache_path, log_q_y_cache_path = _data_cache_paths(
        cache_dir,
        run.id,
        data_key,
    )

    if not force_resample:
        if os.path.exists(sample_cache_path) and os.path.exists(log_q_y_cache_path):
            cached = _load_data_sample_cache(sample_cache_path)
            if cached is not None:
                samples, sample_logps = cached
                log_q_y_means_by_n = _load_log_q_y_mean_cache(log_q_y_cache_path)
                available_n_values = [
                    int(n)
                    for n in n_values
                    if int(n) in log_q_y_means_by_n
                    and int(n) <= int(sample_logps.shape[1])
                ]
                if available_n_values:
                    print(
                        "Using cached data direct MI for hidden_dim="
                        f"{hidden_dim} from {os.path.basename(sample_cache_path)} "
                        f"(num_samples={samples.shape[0]})"
                    )
                    if sanity_check:
                        _sanity_check_direct_data_terms(
                            sample_logps=sample_logps,
                            log_q_y_means_by_n=log_q_y_means_by_n,
                            n_values=available_n_values,
                        )
                    return _compute_bipartite_mi_from_sampled_q(
                        sample_logps=sample_logps,
                        n_values=available_n_values,
                        log_q_y_means_by_n=log_q_y_means_by_n,
                    )
                print(
                    "Skipping hidden_dim="
                    f"{hidden_dim}: cached data direct MI has no usable N values"
                )
                return {}
        print(
            "Skipping hidden_dim=" f"{hidden_dim}: no cached data direct MI available"
        )
        return {}

    import jax

    from training.trainer import create_train_state

    ckpt_path = _download_checkpoint_artifact(run.id, api, cache_dir)
    model = _build_language_model_from_config(cfg)
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

    # Pull the entire cached dataset split (validation by default).
    samples_full = _load_data_chunks(
        cfg=cfg,
        split=data_split,
        num_samples=None,
        rng=None,
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

    log_q_y_means_by_n = _compute_log_q_y_means(
        samples=samples,
        apply_fn=state.apply_fn,
        params=sample_params,
        n_values=n_values,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        progress_desc_prefix=f"Scoring y (data) d_h={hidden_dim}",
        sanity_check=bool(sanity_check),
    )
    _save_log_q_y_mean_cache(log_q_y_cache_path, log_q_y_means_by_n)

    if sanity_check:
        _sanity_check_direct_data_terms(
            sample_logps=sample_logps,
            log_q_y_means_by_n=log_q_y_means_by_n,
            n_values=n_values,
        )

    return _compute_bipartite_mi_from_sampled_q(
        sample_logps=sample_logps,
        n_values=n_values,
        log_q_y_means_by_n=log_q_y_means_by_n,
    )


def _compute_lstm_v_club_for_run_from_data(
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
    seq_len_required = int(n_values[-1])
    seq_len_data = int(cfg.get("seq_len", seq_len_required))
    if seq_len_required > seq_len_data:
        raise RuntimeError(
            "Data sample source requires N <= config.seq_len. "
            f"Got max_n={seq_len_required}, config.seq_len={seq_len_data}"
        )

    data_key = _data_cache_key(
        split=data_split,
        seq_len=seq_len_required,
        num_samples=None,
        bos_token_id=bos_token_id,
        seed=None,
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
            samples, sample_logps = cached
            cached_vclub = _load_estimator_cache(vclub_cache_path)
            available_n_values = [
                int(n)
                for n in n_values
                if int(n) in cached_vclub and int(n) <= int(sample_logps.shape[1])
            ]
            if available_n_values:
                print(
                    "Using cached data v-club for hidden_dim="
                    f"{hidden_dim} from {os.path.basename(vclub_cache_path)}"
                )
                return {int(n): float(cached_vclub[int(n)]) for n in available_n_values}
            print(
                "Skipping hidden_dim="
                f"{hidden_dim}: cached data v-club has no usable N values"
            )
            return {}
        print("Skipping hidden_dim=" f"{hidden_dim}: no cached data v-club available")
        return {}

    import jax

    from training.trainer import create_train_state

    ckpt_path = _download_checkpoint_artifact(run.id, api, cache_dir)
    model = _build_language_model_from_config(cfg)
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
        samples_full = _load_data_chunks(
            cfg=cfg,
            split=data_split,
            num_samples=None,
            rng=None,
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
        int(n) for n in n_values if int(n) <= int(sample_logps.shape[1])
    ]
    if not available_n_values:
        raise RuntimeError(
            f"No usable N values for data v-club hidden_dim={hidden_dim}"
        )

    cached_vclub_values = (
        {} if force_resample else _load_estimator_cache(vclub_cache_path)
    )
    missing_n_values = [
        n for n in available_n_values if int(n) not in cached_vclub_values
    ]
    if missing_n_values:
        computed_vclub_values = _compute_v_club_from_samples(
            samples=samples,
            sample_logps=sample_logps,
            n_values=missing_n_values,
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


def _compute_lstm_v_club_for_run(
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

    reusable_cache = _find_reusable_complete_cache(
        cache_dir=cache_dir,
        run_id=run.id,
        seq_len=n_values[-1],
        target_num_samples=num_samples,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        n_values=n_values,
    )
    if reusable_cache is None and force_resample:
        _compute_lstm_sampled_mi_for_run(
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
            seq_len=n_values[-1],
            target_num_samples=num_samples,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            n_values=n_values,
        )
    if reusable_cache is None:
        print(
            "Skipping hidden_dim=" f"{hidden_dim}: no cached direct artifacts available"
        )
        return {}

    sample_logps = reusable_cache[0]
    sample_cache_path = reusable_cache[3]
    available_n_values = [n for n in n_values if int(n) <= int(sample_logps.shape[1])]
    if not available_n_values:
        raise RuntimeError(f"No usable N values for v-club hidden_dim={hidden_dim}")

    cached_samples = _load_sample_cache(sample_cache_path)
    if cached_samples is None:
        raise RuntimeError(f"Failed to load cached samples from {sample_cache_path}")
    samples, cached_sample_logps = cached_samples

    vclub_cache_path = _estimator_cache_path_for_sample_cache(
        sample_cache_path,
        "v-club",
    )
    cached_vclub_values = _load_estimator_cache(vclub_cache_path)
    missing_n_values = [
        n for n in available_n_values if int(n) not in cached_vclub_values
    ]

    if missing_n_values:
        import jax

        from training.trainer import create_train_state

        ckpt_path = _download_checkpoint_artifact(run.id, api, cache_dir)
        model = _build_language_model_from_config(cfg)
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
                "Checkpoint/run mismatch: "
                f"ckpt_run_id={ckpt_run_id}, run.id={run.id}"
            )
        sample_params = _normalize_params_for_step(
            state.params,
            int(cfg["num_layers"]),
        )

        computed_vclub_values = _compute_v_club_from_samples(
            samples=samples,
            sample_logps=cached_sample_logps,
            n_values=missing_n_values,
            apply_fn=state.apply_fn,
            params=sample_params,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            rng=np.random.default_rng(0),
        )
        cached_vclub_values.update(computed_vclub_values)
        _save_estimator_cache(vclub_cache_path, cached_vclub_values)

    run_vclub_values = {
        int(n): float(cached_vclub_values[int(n)])
        for n in available_n_values
        if int(n) in cached_vclub_values
    }
    if not run_vclub_values:
        raise RuntimeError(
            f"No cached/computed v-club values for hidden_dim={hidden_dim}"
        )
    return run_vclub_values


def _fit_power_law(
    ns: np.ndarray,
    ys: np.ndarray,
    fit_nmin: int,
    fit_nmax: int,
) -> tuple[float, float] | None:
    fit_mask = (ns >= float(fit_nmin)) & np.isfinite(ys) & (ys > 0.0)
    if fit_nmax > 0:
        fit_mask = fit_mask & (ns < float(fit_nmax))
    if int(np.sum(fit_mask)) < 2:
        return None
    x = np.log(ns[fit_mask])
    y = np.log(ys[fit_mask])
    if x.size < 2:
        return None
    power, log_const = np.polyfit(x, y, 1)
    const = float(np.exp(log_const))
    if not np.isfinite(const) or not np.isfinite(power):
        return None
    return const, float(power)


def _load_external_mi_data(source: str) -> tuple[np.ndarray, np.ndarray]:
    source_map = {
        "cagnetta": "cagnetta_mi.csv",
        "kaplan": "kaplan_mi.csv",
        "shengqi": "shengqi_mi.csv",
        "shengi": "shengqi_mi.csv",
    }
    if source not in source_map:
        raise RuntimeError(f"Unsupported external MI source: {source}")

    primary = source_map[source]
    candidates = [primary]
    if source == "shengqi":
        candidates.append("shengi_mi.csv")
    if source == "shengi":
        candidates.append("shengqi_mi.csv")

    data_path = None
    for filename in candidates:
        candidate_path = os.path.join(
            os.path.dirname(__file__), "..", "externalData", filename
        )
        if os.path.exists(candidate_path):
            data_path = candidate_path
            break
    if data_path is None:
        raise FileNotFoundError(
            "Missing external MI file. Looked for: " + ", ".join(candidates)
        )

    data = np.loadtxt(data_path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(
            f"Unexpected external MI data shape for '{source}': {data.shape}"
        )
    if data.shape[1] >= 3:
        n_values = 2.0 * data[:, 0]
        mi_values = data[:, 2]
    else:
        n_values = data[:, 0]
        mi_values = data[:, 1]
    return n_values, mi_values


def _plot_bipartite_mi(
    values: dict[str, dict[tuple[str, int], dict[int, float]]],
    estimators: list[str],
    out_path: str,
    title: str,
    group_labels: dict[str, str],
    include_external: list[str] | None = None,
    fit_nmin: int = FIT_NMIN,
    fit_nmax: int = FIT_NMAX,
) -> None:
    series_keys = sorted(
        {
            series_key
            for estimator_values in values.values()
            for series_key in estimator_values
        }
    )
    if not series_keys:
        raise RuntimeError("No MI values available to plot")

    hidden_dims = sorted({int(hidden_dim) for _, hidden_dim in series_keys})

    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.cm.viridis
    if len(hidden_dims) == 1:
        norm = plt.Normalize(vmin=hidden_dims[0] - 1, vmax=hidden_dims[0] + 1)
    else:
        norm = plt.Normalize(vmin=min(hidden_dims), vmax=max(hidden_dims))

    marker_by_estimator = {
        "direct": "^",
        "v-club": "o",
        "n-gram": "s",
    }
    marker_by_group: dict[str, str] = {}
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for idx, group_name in enumerate(sorted({group for group, _ in series_keys})):
        marker_by_group[group_name] = marker_cycle[idx % len(marker_cycle)]

    fit_params_by_series: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for estimator in estimators:
        estimator_values = values.get(estimator, {})
        for series_key in series_keys:
            group_name, hidden_dim = series_key
            series = estimator_values.get(series_key)
            if not series:
                continue
            ns = np.array(sorted(series.keys()), dtype=float)
            mi = np.array([series[int(n)] for n in ns], dtype=float)
            color = cmap(norm(hidden_dim))
            ax.scatter(
                ns,
                mi,
                color=color,
                edgecolor="black",
                marker=(
                    marker_by_estimator[estimator]
                    if len(marker_by_group) == 1
                    else marker_by_group[group_name]
                ),
                s=40,
                alpha=0.5,
            )

            fit = _fit_power_law(ns, mi, fit_nmin, fit_nmax)
            fit_window_mask = ns >= float(fit_nmin)
            if fit_nmax > 0:
                fit_window_mask = fit_window_mask & (ns < float(fit_nmax))
            if fit is not None and np.any(fit_window_mask):
                const, power = fit
                fit_ns_min = float(np.min(ns[fit_window_mask]))
                fit_ns_max = float(np.max(ns[fit_window_mask]))
                fit_ns = np.logspace(
                    np.log10(fit_ns_min),
                    np.log10(fit_ns_max),
                    num=200,
                )
                fit_curve = const * np.power(fit_ns, power)
                ax.plot(
                    fit_ns,
                    fit_curve,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )
                print(
                    f"fit group={group_name} estimator={estimator} hidden_dim={hidden_dim}: "
                    f"MI={const:.4g}*N^{power:.4f} "
                    f"(N in [{fit_nmin}, {fit_nmax if fit_nmax > 0 else 'inf'}))"
                )
                fit_params_by_series.setdefault(series_key, {})[estimator] = (
                    const,
                    power,
                )

    external_legend_items: list[tuple[str, str, str, bool]] = []
    if include_external:
        ext_colors = ["black", "dimgray", "slategray", "gray"]
        ext_markers = ["x", "+", "1", "2"]
        for ext_idx, external_source in enumerate(include_external):
            ns_external, mi_external = _load_external_mi_data(external_source)
            source_label = {
                "cagnetta": "Cagnetta et al.",
                "kaplan": "Kaplan et al.",
                "shengqi": "Shengqi et al.",
                "shengi": "Shengqi et al.",
            }[external_source]
            order = np.argsort(ns_external)
            ns_external = ns_external[order]
            mi_external = mi_external[order]
            ext_color = ext_colors[ext_idx % len(ext_colors)]
            ext_marker = ext_markers[ext_idx % len(ext_markers)]
            ax.scatter(
                ns_external,
                mi_external,
                color=ext_color,
                edgecolor=ext_color,
                marker=ext_marker,
                s=48,
                alpha=0.5,
            )
            fit_ext = _fit_power_law(ns_external, mi_external, fit_nmin, fit_nmax)
            fit_window_mask = ns_external >= float(fit_nmin)
            if fit_nmax > 0:
                fit_window_mask = fit_window_mask & (ns_external < float(fit_nmax))
            if fit_ext is not None and np.any(fit_window_mask):
                const, power = fit_ext
                fit_ns_min = float(np.min(ns_external[fit_window_mask]))
                fit_ns_max = float(np.max(ns_external[fit_window_mask]))
                fit_ns = np.logspace(
                    np.log10(fit_ns_min),
                    np.log10(fit_ns_max),
                    num=200,
                )
                fit_curve = const * np.power(fit_ns, power)
                ax.plot(
                    fit_ns,
                    fit_curve,
                    color=ext_color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.8,
                )
                external_label = (
                    rf"{source_label}: $I(A:B)={const:.3g}\cdot N^{{{power:.3f}}}$"
                )
                external_legend_items.append(
                    (external_label, ext_color, ext_marker, True)
                )
            else:
                external_legend_items.append(
                    (source_label, ext_color, ext_marker, False)
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$I(A:B)$")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("hidden_dim")

    legend_handles: list[Line2D] = []
    for series_key in series_keys:
        group_name, hidden_dim = series_key
        estimator_fits = fit_params_by_series.get(series_key, {})
        if not estimator_fits:
            continue
        fit_parts: list[str] = []
        for estimator in estimators:
            fit = estimator_fits.get(estimator)
            if fit is None:
                continue
            const, power = fit
            fit_parts.append(
                rf"{estimator}: $I(A:B)={const:.3g}\cdot N^{{{power:.3f}}}$"
            )
        if not fit_parts:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=cmap(norm(hidden_dim)),
                linestyle=":",
                linewidth=1.5,
                label=(
                    (
                        rf"$d_h={hidden_dim}$ [{group_labels.get(group_name, group_name)}], "
                        if group_labels
                        else rf"$d_h={hidden_dim}$, "
                    )
                    + "; ".join(fit_parts)
                ),
            )
        )
    for label, color, marker, has_fit in external_legend_items:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=":" if has_fit else "-",
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        )
    # if legend_handles:
    #     ax.legend(
    #         handles=legend_handles,
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, -0.08),
    #         ncol=1,
    #         frameon=False,
    #     )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved to {out_path}")
    _show_image(out_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, nargs="+", required=True)
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["direct", "v-club", "n-gram"],
        nargs="+",
        default=["direct"],
        help="One or more estimators: direct v-club n-gram",
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
        help="Maximum N to include",
    )
    parser.add_argument(
        "--fit-nmin",
        type=int,
        default=FIT_NMIN,
        help="Min N to include in power-law fits (N >= fit_nmin)",
    )
    parser.add_argument(
        "--fit-nmax",
        type=int,
        default=FIT_NMAX,
        help="Max N to include in power-law fits (N < fit_nmax). Use <=0 for no upper bound",
    )
    parser.add_argument(
        "--sample-source",
        type=str,
        choices=["model", "data"],
        default="model",
        help="Source of sequences for MI: model samples or cached dataset chunks",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        choices=["validation", "train", "test", "validation+test", "test+validation"],
        default="validation",
        help=(
            "Which cached dataset split to draw chunks from when --sample-source=data; "
            "validation+test concatenates both splits"
        ),
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="RNG seed for selecting data chunks when --sample-source=data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of sampled sequences (used only for --sample-source=model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sampling and scoring",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
        help="Directory for MI cache",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Force regeneration of direct caches instead of cache-only mode",
    )
    parser.add_argument(
        "--sanity-check-direct-data",
        action="store_true",
        help=(
            "When --estimator direct and --sample-source data, verify that "
            "E[log q(B|A)] equals -sum L_n computed from the scored data logps, "
            "and that the cached log q(B) scalars are consistent with per-position losses."
        ),
    )
    parser.add_argument(
        "--include-external",
        type=str,
        nargs="+",
        choices=["cagnetta", "kaplan", "shengqi", "shengi"],
        default=None,
        help="Overlay one or more external MI curves from externalData/*_mi.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_n < DEFAULT_MIN_N:
        raise RuntimeError(f"--max-n must be >= {DEFAULT_MIN_N}")
    if args.num_samples < 1:
        raise RuntimeError("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")
    if args.fit_nmin < 1:
        raise RuntimeError("--fit-nmin must be >= 1")
    if bool(getattr(args, "sanity_check_direct_data", False)):
        if "direct" not in (args.estimator or []):
            raise RuntimeError("--sanity-check-direct-data requires --estimator direct")
        if str(getattr(args, "sample_source", "model")) != "data":
            raise RuntimeError(
                "--sanity-check-direct-data requires --sample-source data"
            )


def _dedupe_estimators(estimator_args: list[str]) -> list[str]:
    estimators: list[str] = []
    for estimator in estimator_args:
        if estimator not in estimators:
            estimators.append(estimator)
    return estimators


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)
    estimators = _dedupe_estimators(args.estimator)

    n_values = [n for n in DEFAULT_N_VALUES if n <= int(args.max_n)]
    if not n_values:
        raise RuntimeError("No valid N values to evaluate")

    groups = list(dict.fromkeys(args.group))
    group_labels = distinct_group_labels(groups) if len(groups) > 1 else {}

    api = wandb.Api()
    all_values: dict[str, dict[tuple[str, int], dict[int, float]]] = {
        estimator: {} for estimator in estimators
    }
    for group_name in groups:
        runs = _resolve_group_runs(api, group_name)
        runs = _filter_runs_by_hidden_dim(runs, args.hidden_dim, group_name)
        for run in tqdm(runs, desc=f"Runs[{group_name}]", unit="run"):
            cfg = run.config or {}
            hidden_dim = int(cfg["hidden_dim"])
            series_key = (group_name, hidden_dim)

            if "direct" in estimators:
                if args.sample_source == "data":
                    direct_values = _compute_lstm_direct_mi_for_run_from_data(
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
                        sanity_check=bool(args.sanity_check_direct_data),
                    )
                    if direct_values:
                        all_values["direct"][series_key] = direct_values
                else:
                    direct_values = _compute_lstm_sampled_mi_for_run(
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
                    if direct_values:
                        all_values["direct"][series_key] = direct_values

            if "v-club" in estimators:
                if args.sample_source == "data":
                    vclub_values = _compute_lstm_v_club_for_run_from_data(
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
                    if vclub_values:
                        all_values["v-club"][series_key] = vclub_values
                else:
                    vclub_values = _compute_lstm_v_club_for_run(
                        run,
                        api,
                        hidden_dim,
                        n_values,
                        num_samples=args.num_samples,
                        batch_size=args.batch_size,
                        cache_dir=args.cache_dir,
                        force_resample=args.force_resample,
                    )
                    if vclub_values:
                        all_values["v-club"][series_key] = vclub_values

            if "n-gram" in estimators:
                all_values["n-gram"][series_key] = _compute_ngram_mi_for_run(
                    run,
                    n_values,
                )

    out_path = (
        args.output
        if args.output is not None
        else f"results/bipartite_mi_{'_'.join(groups)}.png"
    )
    _plot_bipartite_mi(
        all_values,
        estimators,
        out_path,
        title=(
            "Bipartite MI " f"({', '.join(estimators)}, groups={', '.join(groups)})"
        ),
        group_labels=group_labels,
        include_external=(
            list(dict.fromkeys(args.include_external))
            if args.include_external
            else None
        ),
        fit_nmin=args.fit_nmin,
        fit_nmax=args.fit_nmax,
    )


if __name__ == "__main__":
    main()

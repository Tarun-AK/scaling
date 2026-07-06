from __future__ import annotations

import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import wandb
from analysis.plot_bipartite_mi import (
    _build_language_model_from_config,
    _download_checkpoint_artifact,
    _load_checkpoint,
    _normalize_params_for_step,
    _resolve_group_runs,
)
from data.dataset import load_splits_as_arrays
from training.trainer import create_train_state


def _prepend_bos(tokens: jax.Array, bos_token_id: int) -> jax.Array:
    bos = jnp.full((tokens.shape[0], 1), bos_token_id, dtype=tokens.dtype)
    return jnp.concatenate([bos, tokens[:, :-1]], axis=1)


def _score_half(
    *,
    tokens_half: np.ndarray,
    apply_fn,
    params: dict,
    bos_token_id: int,
    batch_size: int,
) -> tuple[float, float]:
    n_seq, half_len = tokens_half.shape
    sum_logp_total = 0.0
    sum_logp_by_pos = np.zeros((half_len,), dtype=np.float64)

    @jax.jit
    def batch_token_logp(batch_tokens: jax.Array) -> jax.Array:
        inputs = _prepend_bos(batch_tokens, bos_token_id)
        logits = apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(
            log_probs,
            batch_tokens[:, :, None],
            axis=-1,
        ).squeeze(-1)
        return token_logp

    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        batch = jnp.asarray(tokens_half[start:end], dtype=jnp.int32)
        token_logp = np.asarray(batch_token_logp(batch), dtype=np.float64)
        sum_logp_total += float(np.sum(token_logp))
        sum_logp_by_pos += np.sum(token_logp, axis=0)

    avg_log_x = sum_logp_total / float(n_seq)
    l_by_pos = -sum_logp_by_pos / float(n_seq)
    avg_ngram = float(np.mean(l_by_pos))

    lhs = float(avg_log_x)
    rhs = float(-half_len * avg_ngram)
    if not np.isclose(lhs, rhs, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "Sanity check failed for half: "
            f"avg_log={lhs:.12f}, -len*avg_ngram={rhs:.12f}"
        )

    return avg_log_x, avg_ngram


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check avg log p(x_A/x_B) and avg n-gram losses on validation",
    )
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints/bipartite_mi_cache",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = _resolve_group_runs(api, args.group)
    run = None
    for candidate in runs:
        hidden_dim = int((candidate.config or {}).get("hidden_dim", -1))
        if hidden_dim == int(args.hidden_dim):
            run = candidate
            break

    if run is None:
        raise RuntimeError(
            f"No finished run for group='{args.group}' hidden_dim={args.hidden_dim}"
        )

    cfg = run.config or {}
    bos_token_id = int(cfg.get("bos_token_id", 0))
    seq_len = int(cfg["seq_len"])
    if seq_len % 2 != 0:
        raise RuntimeError(f"seq_len must be even, got {seq_len}")

    _, val_np, _ = load_splits_as_arrays(
        dataset_name=str(cfg["dataset_name"]),
        dataset_config=cfg.get("dataset_config"),
        seq_len=seq_len,
        vocab_size=int(cfg["vocab_size"]),
        cache_dir=str(cfg.get("cache_dir", "data/cache")),
        require_cache=bool(cfg.get("require_cached_data", True)),
        tokenize_batch_size=int(cfg.get("tokenize_batch_size", 32)),
        tokenizer_path=str(cfg.get("tokenizer_path", "data/tokenizer/tokenizer.json")),
        dataset_path=(
            str(cfg.get("dataset_path"))
            if cfg.get("dataset_path") is not None
            else None
        ),
    )

    val_tokens = np.asarray(val_np, dtype=np.int32)
    half = seq_len // 2
    x_a = val_tokens[:, :half]
    x_b = val_tokens[:, half:]

    model = _build_language_model_from_config(cfg)
    state_cfg = SimpleNamespace(
        batch_size=int(cfg.get("batch_size", 1)),
        seq_len=seq_len,
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )
    state = create_train_state(model, state_cfg, jax.random.PRNGKey(0))
    ckpt_path = _download_checkpoint_artifact(run.id, api, args.cache_dir)
    state, restored = _load_checkpoint(ckpt_path, state)

    ckpt_run_id = restored.get("wandb_run_id")
    if ckpt_run_id != run.id:
        raise RuntimeError(
            f"Checkpoint/run mismatch: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
        )

    params = _normalize_params_for_step(state.params, int(cfg["num_layers"]))

    avg_log_b, avg_ngram_b = _score_half(
        tokens_half=x_b,
        apply_fn=state.apply_fn,
        params=params,
        bos_token_id=bos_token_id,
        batch_size=int(args.batch_size),
    )
    avg_log_a, avg_ngram_a = _score_half(
        tokens_half=x_a,
        apply_fn=state.apply_fn,
        params=params,
        bos_token_id=bos_token_id,
        batch_size=int(args.batch_size),
    )

    print(f"{avg_log_b:.12f}")
    print(f"{avg_ngram_b:.12f}")
    print(f"{avg_log_a:.12f}")
    print(f"{avg_ngram_a:.12f}")


if __name__ == "__main__":
    main()

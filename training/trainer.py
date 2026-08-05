"""Training loop for LSTM scaling experiments.

Key features:
- JIT-compiled train/eval steps.
- Weights & Biases logging.
- In-epoch test-loss monitoring with optional early stopping.
- Orbax checkpointing keyed by hidden_dim.

Notes on JAX gotchas:
- Anything that changes shapes should be kept consistent to avoid recompilation.
  We therefore drop the last partial batch for train/val/test.
- PRNG keys are managed explicitly for parameter initialization.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader import batch_iterator
from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from models.mamba import MambaLanguageModel
from models.transformer import TransformerLanguageModel
from models.vanilla_rnn import VanillaRNNLanguageModel
from training.loss import cross_entropy_loss
from training.metrics import compute_all_ngram_losses


class TrainState(train_state.TrainState):
    """Flax TrainState with no extra fields."""


def create_train_state(
    model: (
        LSTMLanguageModel
        | VanillaRNNLanguageModel
        | MambaLanguageModel
        | TransformerLanguageModel
    ),
    config: Any,
    rng: jax.Array,
    tx: optax.GradientTransformation | None = None,
) -> TrainState:
    """Create an initialized TrainState."""
    # Use the smallest practical dummy input for parameter initialization.
    # The parameter shapes do not depend on the training batch size, and
    # avoiding a full batch here keeps very long sequence runs from blowing up
    # host memory during init/compile.
    dummy_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    params = model.init(rng, dummy_tokens)["params"]
    if tx is None:
        tx = optax.adam(learning_rate=float(config.learning_rate))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _build_optimizer(config: Any, total_train_steps: int) -> optax.GradientTransformation:
    learning_rate = float(config.learning_rate)
    schedule_type = str(getattr(config, "lr_schedule", "constant")).lower()
    if schedule_type == "constant":
        return optax.adam(learning_rate=learning_rate)

    if schedule_type != "warmup_cosine":
        raise RuntimeError(
            "Unsupported lr_schedule. Expected one of ['constant', 'warmup_cosine'], "
            f"got '{schedule_type}'"
        )

    warmup_steps = int(getattr(config, "warmup_steps", 0))
    min_learning_rate = float(getattr(config, "min_learning_rate", 0.0))
    if warmup_steps < 0:
        raise RuntimeError("warmup_steps must be >= 0")
    if min_learning_rate < 0.0:
        raise RuntimeError("min_learning_rate must be >= 0")
    if min_learning_rate > learning_rate:
        raise RuntimeError("min_learning_rate must be <= learning_rate")

    warmup_steps_eff = min(warmup_steps, max(0, int(total_train_steps) - 1))
    decay_steps = max(1, int(total_train_steps) - warmup_steps_eff)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps_eff,
        decay_steps=decay_steps,
        end_value=min_learning_rate,
    )
    return optax.adam(learning_rate=lr_schedule)


def _prepend_bos(tokens: jax.Array, bos_token_id: int) -> jax.Array:
    bos = jnp.full((tokens.shape[0], 1), bos_token_id, dtype=tokens.dtype)
    return jnp.concatenate([bos, tokens[:, :-1]], axis=1)


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


@functools.partial(
    jax.jit,
    static_argnames=("bos_token_id", "grad_accum_steps"),
    donate_argnums=(0,),
)
def _train_step_jit(
    state: TrainState,
    batch: jax.Array,
    bos_token_id: int,
    grad_accum_steps: int,
) -> Tuple[TrainState, jax.Array]:
    """One optimizer step, entirely inside a single jit.

    The accumulation loop, the gradient averaging and the optimizer update are
    all traced together: done outside jit they dispatch a separate kernel per
    parameter per microbatch, which dominates the step for small models. The
    state is donated so params and Adam moments are updated in place.
    """

    def microbatch_grads(params, micro_batch):
        def loss_fn(p):
            logits = state.apply_fn(
                {"params": p}, _prepend_bos(micro_batch, bos_token_id)
            )
            return cross_entropy_loss(logits, micro_batch)

        return jax.value_and_grad(loss_fn)(params)

    if grad_accum_steps == 1:
        loss, grads = microbatch_grads(state.params, batch)
    else:
        micro_batches = batch.reshape(
            grad_accum_steps, batch.shape[0] // grad_accum_steps, batch.shape[1]
        )

        def accumulate(carry, micro_batch):
            grads_sum, loss_sum = carry
            loss, grads = microbatch_grads(state.params, micro_batch)
            return (
                jax.tree_util.tree_map(jnp.add, grads_sum, grads),
                loss_sum + loss,
            ), None

        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        (grads_sum, loss_sum), _ = jax.lax.scan(
            accumulate, (zero_grads, jnp.zeros((), jnp.float32)), micro_batches
        )
        grads = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, grads_sum)
        loss = loss_sum / grad_accum_steps

    return state.apply_gradients(grads=grads), loss


def train_step(
    state: TrainState,
    batch: jax.Array,
    rng: jax.Array,
    bos_token_id: int,
    grad_accum_steps: int,
) -> Tuple[TrainState, Dict[str, jax.Array], jax.Array]:
    """Perform one optimizer step, optionally with gradient accumulation.

    NOTE: `state` is donated, so the caller must rebind it from the return value
    and must not touch the state it passed in.
    """
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps < 1:
        raise RuntimeError("grad_accum_steps must be >= 1")

    batch_size = int(batch.shape[0])
    if batch_size % grad_accum_steps != 0:
        raise RuntimeError(
            f"batch_size={batch_size} must be divisible by "
            f"grad_accum_steps={grad_accum_steps}"
        )

    new_state, loss = _train_step_jit(state, batch, bos_token_id, grad_accum_steps)
    return new_state, {"loss": loss}, rng


@functools.partial(jax.jit, static_argnames=("bos_token_id",))
def eval_step(
    state: TrainState,
    batch: jax.Array,
    bos_token_id: int,
) -> jax.Array:
    """Compute all n-gram losses for a batch."""
    return compute_all_ngram_losses(state.apply_fn, state.params, batch, bos_token_id)


@functools.partial(jax.jit, static_argnames=("bos_token_id",))
def eval_loss_step(
    state: TrainState,
    batch: jax.Array,
    bos_token_id: int,
) -> jax.Array:
    inputs = _prepend_bos(batch, bos_token_id)
    logits = state.apply_fn({"params": state.params}, inputs)
    return cross_entropy_loss(logits, batch)


def _mean_metrics(metrics_list: Iterable[jax.Array]) -> Dict[str, float]:
    """Average a list of per-position-loss arrays into a dict of Python floats.

    Each element of metrics_list is pulled off device with a single
    jax.device_get call (not one per sequence position) -- avoids seq_len
    blocking host<->device round trips per batch.
    """
    total = None
    count = 0
    for m in metrics_list:
        count += 1
        values = np.asarray(jax.device_get(m))
        total = values if total is None else total + values
    if count == 0:
        return {}
    mean = total / count
    return {f"ngram_{n}": float(v) for n, v in enumerate(mean, start=1)}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _config_to_dict(config: Any) -> dict:
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    try:
        return dict(config)
    except TypeError:
        return vars(config)


def _eval_split(
    state: TrainState, data: np.ndarray, batch_size: int, bos_token_id: int
) -> Dict[str, float]:
    return _mean_metrics(
        eval_step(state, batch, bos_token_id)
        for batch in batch_iterator(
            data,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            drop_last=True,
        )
    )


def _eval_loss(
    state: TrainState,
    data: np.ndarray,
    batch_size: int,
    bos_token_id: int,
) -> float:
    total = 0.0
    count = 0
    for batch in batch_iterator(
        data,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        drop_last=True,
    ):
        total += float(jax.device_get(eval_loss_step(state, batch, bos_token_id)))
        count += 1
    if count == 0:
        return float("inf")
    return total / count


def _sample_sequences(
    model: (
        LSTMLanguageModel
        | VanillaRNNLanguageModel
        | MambaLanguageModel
        | TransformerLanguageModel
    ),
    params: dict,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
    rng: jax.Array,
) -> np.ndarray:
    def _sample_batch(
        batch_rng: jax.Array, init_carry: tuple, seq_len: int, bos_token_id: int
    ) -> jax.Array:
        def scan_step(state, _):
            lstm_carry, logits = model.apply(
                {"params": params},
                state["carry"],
                state["token"],
                method=model.step,
            )
            rng, step_rng = jax.random.split(state["rng"])
            next_token = jax.random.categorical(step_rng, logits)
            new_state = {"carry": lstm_carry, "token": next_token, "rng": rng}
            return new_state, next_token

        batch_size_local = int(jax.tree_util.tree_leaves(init_carry)[0].shape[0])
        init_token = jnp.full((batch_size_local,), bos_token_id)
        carry = {"carry": init_carry, "token": init_token, "rng": batch_rng}
        _, tokens = jax.lax.scan(scan_step, carry, jnp.arange(seq_len))
        return jnp.transpose(tokens, (1, 0))

    sample_batch_jit = jax.jit(
        _sample_batch, static_argnames=("seq_len", "bos_token_id")
    )

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        rng, batch_rng = jax.random.split(rng)
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        init_carry = model.init_carry(current_batch_size)
        tokens = sample_batch_jit(batch_rng, init_carry, seq_len, bos_token_id)
        all_samples.append(np.array(tokens))

    return np.concatenate(all_samples, axis=0)[:num_samples]


def _compute_conditional_entropy_from_samples(
    apply_fn,
    params,
    samples: np.ndarray,
    batch_size: int,
    bos_token_id: int,
) -> Dict[str, float]:
    all_metrics = []
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(samples))
        batch = jnp.array(samples[start:end])
        inputs = _prepend_bos(batch, bos_token_id)
        targets = batch

        logits = apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)

        per_position = -jnp.mean(token_logp, axis=0)
        all_metrics.append(np.array(per_position))

    if not all_metrics:
        return {}

    mean_per_position = np.mean(np.stack(all_metrics), axis=0)
    return {
        f"entropy_{n + 1}": float(mean_per_position[n])
        for n in range(len(mean_per_position))
    }


def _checkpoint_metadata(
    config: Any, epoch: float | int | None, tokens_seen: int | None = None
) -> dict[str, int | float | str]:
    metadata: dict[str, int | float | str] = {
        "model_type": str(getattr(config, "model_type", "lstm")),
        "hidden_dim": int(config.hidden_dim),
        "num_layers": int(config.num_layers),
        "seq_len": int(config.seq_len),
        "vocab_size": int(config.vocab_size),
    }
    if epoch is not None:
        metadata["epoch"] = epoch
    if tokens_seen is not None:
        metadata["tokens_seen"] = int(tokens_seen)
    return metadata


def _save_checkpoint(
    *,
    state: TrainState,
    config: Any,
    checkpoint_root: str,
    wandb_run_id: str,
    epoch: float | int | None,
    tokens_seen: int | None = None,
) -> str:
    _ensure_dir(checkpoint_root)
    checkpointer = ocp.PyTreeCheckpointer()
    payload = {
        "params": state.params,
        "config": _config_to_dict(config),
        "wandb_run_id": wandb_run_id,
        "epoch": epoch,
    }
    if tokens_seen is not None:
        payload["tokens_seen"] = int(tokens_seen)
    checkpointer.save(
        os.path.join(checkpoint_root, "ckpt"),
        payload,
        force=True,
    )
    return checkpoint_root


def _log_checkpoint_artifact(
    *,
    config: Any,
    checkpoint_root: str,
    artifact_name: str,
    wandb_run_id: str,
    epoch: float | int | None,
    tokens_seen: int | None = None,
) -> None:
    import wandb

    metadata = {
        "wandb_run_id": wandb_run_id,
        **_checkpoint_metadata(config, epoch, tokens_seen),
    }
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata,
    )
    artifact.add_dir(checkpoint_root)
    wandb.log_artifact(artifact)


def _save_epoch_checkpoint(
    *,
    state: TrainState,
    config: Any,
    epoch_label: str,
    epoch_value: float | int,
    wandb_run_id: str,
) -> None:
    """Save + log a checkpoint tagged with a (possibly fractional) epoch label.

    epoch_label controls the on-disk directory / artifact name (e.g. "0000.5"
    for the halfway point of the first epoch, "0001" for its end);
    epoch_value is the numeric epoch stored in checkpoint/artifact metadata.
    """
    checkpoint_root = os.path.join(
        str(config.checkpoint_dir),
        "epochs",
        f"hidden_dim={int(config.hidden_dim)}",
        f"epoch={epoch_label}",
    )
    _save_checkpoint(
        state=state,
        config=config,
        checkpoint_root=checkpoint_root,
        wandb_run_id=wandb_run_id,
        epoch=epoch_value,
    )
    _log_checkpoint_artifact(
        config=config,
        checkpoint_root=checkpoint_root,
        artifact_name=f"checkpoint-{wandb_run_id}-epoch-{epoch_label}",
        wandb_run_id=wandb_run_id,
        epoch=epoch_value,
    )


# Width of the zero-padded token count in token-checkpoint artifact names.
# Zero padding keeps them in training order under lexicographic sort, and 13
# digits covers up to 10^13 tokens. analysis/plot_bipartite_mi.py mirrors this.
TOKEN_LABEL_DIGITS = 13


def token_checkpoint_label(tokens: int) -> str:
    """Artifact-name label for a checkpoint taken at ``tokens`` tokens seen."""
    return f"{int(tokens):0{TOKEN_LABEL_DIGITS}d}"


def _save_token_checkpoint(
    *,
    state: TrainState,
    config: Any,
    nominal_tokens: int,
    tokens_seen: int,
    epoch_value: float,
    wandb_run_id: str,
) -> None:
    """Save + log a checkpoint tagged by cumulative training tokens.

    ``nominal_tokens`` is the milestone the checkpoint stands for (a multiple of
    checkpoint_every_n_tokens) and names the artifact; ``tokens_seen`` is the
    exact count, which overshoots the milestone by up to one batch and is kept
    in metadata. Naming by the milestone is what makes the same checkpoint index
    line up across runs with different batch sizes or sequence lengths.
    """
    label = token_checkpoint_label(nominal_tokens)
    checkpoint_root = os.path.join(
        str(config.checkpoint_dir),
        "tokens",
        f"hidden_dim={int(config.hidden_dim)}",
        f"tokens={label}",
    )
    _save_checkpoint(
        state=state,
        config=config,
        checkpoint_root=checkpoint_root,
        wandb_run_id=wandb_run_id,
        epoch=epoch_value,
        tokens_seen=tokens_seen,
    )
    _log_checkpoint_artifact(
        config=config,
        checkpoint_root=checkpoint_root,
        artifact_name=f"checkpoint-{wandb_run_id}-tokens-{label}",
        wandb_run_id=wandb_run_id,
        epoch=epoch_value,
        tokens_seen=tokens_seen,
    )


def _post_training_evals_and_checkpoint(
    *,
    state: TrainState,
    model: (
        LSTMLanguageModel
        | VanillaRNNLanguageModel
        | MambaLanguageModel
        | TransformerLanguageModel
    ),
    config: Any,
    rng: jax.Array,
    train_np: np.ndarray,
    val_np: np.ndarray,
    test_np: np.ndarray,
    global_step: int,
    final_epoch: int,
    early_stopped: bool,
    best_test_loss: float | None,
    best_test_step: int | None,
) -> Dict[str, float]:
    import wandb

    bos_token_id = int(getattr(config, "bos_token_id", 0))
    eval_batch_size = int(getattr(config, "eval_batch_size", config.batch_size))

    final_test_loss = _eval_loss(state, test_np, eval_batch_size, bos_token_id)
    wandb.log({"test/loss": final_test_loss}, step=global_step)

    test_mean = _eval_split(state, test_np, eval_batch_size, bos_token_id)
    wandb.log({f"test/{k}": v for k, v in test_mean.items()}, step=global_step)

    combined_np = np.concatenate([val_np, test_np], axis=0)
    combined_mean = _eval_split(state, combined_np, eval_batch_size, bos_token_id)
    wandb.log({f"combined/{k}": v for k, v in combined_mean.items()})

    if bool(getattr(config, "log_train_ngram_after_training", False)):
        train_mean = _eval_split(state, train_np, eval_batch_size, bos_token_id)
        wandb.log({f"train_ngram/{k}": v for k, v in train_mean.items()})

    entropy_num_samples = int(getattr(config, "entropy_num_samples", 0))
    entropy_batch_size = int(getattr(config, "entropy_sample_batch_size", 0))
    if entropy_num_samples > 0 and entropy_batch_size > 0:
        try:
            rng, entropy_rng = jax.random.split(rng)
            sample_params = _normalize_params_for_step(state.params, int(config.num_layers))
            samples = _sample_sequences(
                model=model,
                params=sample_params,
                seq_len=int(config.seq_len),
                num_samples=entropy_num_samples,
                batch_size=entropy_batch_size,
                bos_token_id=bos_token_id,
                rng=entropy_rng,
            )
            entropy_dict = _compute_conditional_entropy_from_samples(
                apply_fn=state.apply_fn,
                params=state.params,
                samples=samples,
                batch_size=entropy_batch_size,
                bos_token_id=bos_token_id,
            )
            if entropy_dict:
                wandb.log(
                    {f"conditional_entropy/{k}": v for k, v in entropy_dict.items()}
                )
        except Exception as exc:
            print(f"Warning: skipping conditional entropy post-eval due to error: {exc}")
            wandb.log({"conditional_entropy/skipped": 1})

    final_checkpoint_root = os.path.join(
        str(config.checkpoint_dir), f"hidden_dim={int(config.hidden_dim)}"
    )
    _save_checkpoint(
        state=state,
        config=config,
        checkpoint_root=final_checkpoint_root,
        wandb_run_id=wandb.run.id,
        epoch=final_epoch,
    )
    _log_checkpoint_artifact(
        config=config,
        checkpoint_root=final_checkpoint_root,
        artifact_name=f"checkpoint-{wandb.run.id}",
        wandb_run_id=wandb.run.id,
        epoch=final_epoch,
    )

    summary_payload = {
        "training/early_stopped": int(early_stopped),
        "training/final_epoch": final_epoch,
    }
    if best_test_loss is not None:
        summary_payload["training/best_test_loss"] = best_test_loss
    if best_test_step is not None:
        summary_payload["training/best_test_step"] = best_test_step
    wandb.log(summary_payload, step=global_step)

    wandb.finish()
    return test_mean


def _wandb_init(config: Any) -> None:
    import wandb

    cfg_dict = _config_to_dict(config)
    wandb.init(
        project=str(getattr(config, "wandb_project", "scaling")),
        entity=getattr(config, "wandb_entity", None),
        group=getattr(config, "wandb_group", None),
        name=getattr(config, "run_name", None),
        config=cfg_dict,
    )


def train_and_evaluate(config: Any) -> Dict[str, float]:
    """Run training loop, final test evaluation, and checkpointing."""
    _ensure_dir(str(config.checkpoint_dir))
    _ensure_dir(str(config.results_dir))
    bos_token_id = int(getattr(config, "bos_token_id", 0))
    dataset_config = getattr(config, "dataset_config", None)
    dataset_path = getattr(config, "dataset_path", None)
    tokenizer_path = str(
        getattr(config, "tokenizer_path", "data/tokenizer/tokenizer.json")
    )

    # Data
    train_np, val_np, test_np = load_splits_as_arrays(
        dataset_name=str(config.dataset_name),
        dataset_config=dataset_config,
        seq_len=int(config.seq_len),
        vocab_size=int(config.vocab_size),
        cache_dir=str(getattr(config, "cache_dir", "data/cache")),
        require_cache=bool(getattr(config, "require_cached_data", True)),
        tokenize_batch_size=int(getattr(config, "tokenize_batch_size", 32)),
        tokenizer_path=tokenizer_path,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
    )

    # Model/state
    model_type = str(getattr(config, "model_type", "lstm")).lower()
    if model_type == "lstm":
        model = LSTMLanguageModel(
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            vocab_size=int(config.vocab_size),
        )
    elif model_type == "mamba":
        model = MambaLanguageModel(
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            vocab_size=int(config.vocab_size),
            state_size=int(
                getattr(config, "mamba_state_size", getattr(config, "mamba_d_state", 16))
            ),
            head_dim=int(getattr(config, "mamba_head_dim", 16)),
            # 256 was the old fallback and is ~1.6x slower than 16 at
            # hd128/seq8192; see configs/base.yaml:mamba_chunk_size.
            chunk_size=int(getattr(config, "mamba_chunk_size", 16)),
            expand=int(getattr(config, "mamba_expand", 2)),
            conv_kernel=int(getattr(config, "mamba_conv_kernel", 4)),
        )
    elif model_type == "vanilla_rnn":
        model = VanillaRNNLanguageModel(
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            vocab_size=int(config.vocab_size),
        )
    elif model_type == "transformer":
        model = TransformerLanguageModel(
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            vocab_size=int(config.vocab_size),
            num_heads=int(getattr(config, "transformer_num_heads", 8)),
            num_kv_heads=int(
                getattr(
                    config,
                    "transformer_num_kv_heads",
                    getattr(config, "transformer_num_heads", 8),
                )
            ),
            ffn_mult=float(getattr(config, "transformer_ffn_mult", 8.0 / 3.0)),
            rope_theta=float(getattr(config, "transformer_rope_theta", 1_000_000.0)),
            layer_norm_epsilon=float(
                getattr(config, "transformer_layer_norm_epsilon", 1e-6)
            ),
            tie_word_embeddings=bool(
                getattr(config, "transformer_tie_word_embeddings", True)
            ),
            attention_impl=(
                str(_impl)
                if (_impl := getattr(config, "transformer_attention_impl", None))
                and str(_impl).lower() not in {"none", "null", ""}
                else None
            ),
        )
    else:
        raise RuntimeError(
            "Unsupported model_type. Expected one of: "
            "['lstm', 'mamba', 'vanilla_rnn', 'transformer'], got "
            f"'{model_type}'"
        )
    rng = jax.random.PRNGKey(0)
    init_rng = jax.random.PRNGKey(0)

    batch_size = int(config.batch_size)
    grad_accum_steps = int(getattr(config, "grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise RuntimeError("grad_accum_steps must be >= 1")
    if batch_size % grad_accum_steps != 0:
        raise RuntimeError(
            f"batch_size={batch_size} must be divisible by grad_accum_steps={grad_accum_steps}"
        )
    seq_len = int(config.seq_len)
    num_train_batches = train_np.shape[0] // batch_size
    num_epochs = int(config.num_epochs)
    total_train_steps = max(1, num_train_batches * num_epochs)

    tx = _build_optimizer(config, total_train_steps)
    state = create_train_state(model, config, init_rng, tx=tx)
    _wandb_init(config)
    import wandb

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    wandb.config.update({"num_params": num_params})

    if bool(getattr(config, "dry_run", False)):
        first_batch = next(
            batch_iterator(
                train_np,
                batch_size=int(config.batch_size),
                shuffle=False,
                seed=0,
                drop_last=True,
            ),
            None,
        )
        if first_batch is not None:
            state, metrics, rng = train_step(
                state,
                first_batch,
                rng,
                bos_token_id,
                grad_accum_steps,
            )
            _ = eval_loss_step(state, first_batch, bos_token_id)
            wandb.log(
                {
                    "dry_run": 1,
                    "num_params": num_params,
                    "dry_run/train_loss": float(jax.device_get(metrics["loss"])),
                },
                step=0,
            )
        else:
            wandb.log(
                {"dry_run": 1, "num_params": num_params, "dry_run/no_batch": 1},
                step=0,
            )
        wandb.finish()
        return {"dry_run": 1.0, "num_params": float(num_params)}

    # Initial validation before any training
    eval_batch_size = int(getattr(config, "eval_batch_size", config.batch_size))
    initial_val_mean = _eval_split(state, val_np, eval_batch_size, bos_token_id)
    wandb.log({f"val/{k}": v for k, v in initial_val_mean.items()}, step=0)

    eval_every_n_steps = int(getattr(config, "eval_every_n_steps", 0))
    early_stop_patience = int(getattr(config, "early_stop_patience", 0))
    early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 0.0))
    best_test_loss: float | None = None
    best_test_step: int | None = None
    plateau_count = 0

    if eval_every_n_steps > 0:
        best_test_loss = _eval_loss(state, test_np, eval_batch_size, bos_token_id)
        best_test_step = 0
        wandb.log({"test/loss": best_test_loss}, step=0)

    wandb.log({"epoch": 0}, step=0)

    global_step = 0
    early_stopped = False
    final_epoch = 0

    # Checkpoint cadence. Two mutually exclusive modes:
    #  - checkpoint_every_n_tokens > 0: checkpoint on cumulative-token
    #    milestones (plus one at the very end). Token counts are comparable
    #    across corpus sizes, which epoch fractions are not.
    #  - otherwise: the original half-epoch + end-of-epoch cadence.
    checkpoint_every_n_tokens = int(
        getattr(config, "checkpoint_every_n_tokens", 0) or 0
    )
    if checkpoint_every_n_tokens < 0:
        raise RuntimeError("checkpoint_every_n_tokens must be >= 0")
    token_checkpoints = checkpoint_every_n_tokens > 0
    tokens_per_batch = batch_size * seq_len
    tokens_seen = 0
    next_token_milestone = checkpoint_every_n_tokens if token_checkpoints else 0
    last_token_checkpoint = 0

    # Batch index (1-indexed, within the current epoch) at which to save the
    # halfway-point checkpoint, in addition to the existing end-of-epoch one.
    halfway_batch_count = 0 if token_checkpoints else num_train_batches // 2

    if token_checkpoints:
        total_tokens = num_train_batches * num_epochs * tokens_per_batch
        print(
            f"Token checkpointing every {checkpoint_every_n_tokens:,} tokens "
            f"({tokens_per_batch:,} tokens/batch, {total_tokens:,} total => "
            f"{total_tokens // checkpoint_every_n_tokens} milestone checkpoints "
            "+ 1 final)"
        )

    for epoch in range(num_epochs):
        final_epoch = epoch + 1
        train_batches = batch_iterator(
            train_np,
            batch_size=batch_size,
            shuffle=True,
            seed=epoch,
            drop_last=True,
        )
        progress = tqdm(
            train_batches,
            total=num_train_batches,
            desc=f"epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )
        batches_seen_this_epoch = 0
        for batch in progress:
            state, metrics, rng = train_step(
                state,
                batch,
                rng,
                bos_token_id,
                grad_accum_steps,
            )
            global_step += 1
            batches_seen_this_epoch += 1
            tokens_seen += tokens_per_batch
            if (
                halfway_batch_count > 0
                and batches_seen_this_epoch == halfway_batch_count
            ):
                _save_epoch_checkpoint(
                    state=state,
                    config=config,
                    epoch_label=f"{epoch:04d}.5",
                    epoch_value=epoch + 0.5,
                    wandb_run_id=wandb.run.id,
                )
            if token_checkpoints and tokens_seen >= next_token_milestone:
                _save_token_checkpoint(
                    state=state,
                    config=config,
                    nominal_tokens=next_token_milestone,
                    tokens_seen=tokens_seen,
                    epoch_value=epoch + batches_seen_this_epoch / num_train_batches,
                    wandb_run_id=wandb.run.id,
                )
                last_token_checkpoint = next_token_milestone
                # A single batch can only cross one milestone in practice, but
                # advance past every crossed one so the cadence cannot drift.
                while next_token_milestone <= tokens_seen:
                    next_token_milestone += checkpoint_every_n_tokens
            rate = progress.format_dict.get("rate", None)
            if rate is not None:
                progress.set_postfix(
                    {
                        "it/s": f"{rate:.2f}",
                        "tok/s": f"{rate * batch_size * seq_len:,.0f}",
                    },
                    refresh=False,
                )
            if global_step % int(config.log_every_n_steps) == 0:
                wandb.log(
                    {
                        f"train/{k}": float(jax.device_get(v))
                        for k, v in metrics.items()
                    },
                    step=global_step,
                )

            if eval_every_n_steps > 0 and global_step % eval_every_n_steps == 0:
                current_test_loss = _eval_loss(
                    state,
                    test_np,
                    eval_batch_size,
                    bos_token_id,
                )
                wandb.log({"test/loss": current_test_loss}, step=global_step)

                if early_stop_patience > 0:
                    if best_test_loss is None or current_test_loss < (
                        best_test_loss - early_stop_min_delta
                    ):
                        best_test_loss = current_test_loss
                        best_test_step = global_step
                        plateau_count = 0
                    else:
                        plateau_count += 1
                        if plateau_count >= early_stop_patience:
                            early_stopped = True
                            break

        progress.close()

        if early_stopped:
            break

        # Validation at end of each epoch
        val_mean = _eval_split(state, val_np, eval_batch_size, bos_token_id)
        wandb.log({f"val/{k}": v for k, v in val_mean.items()}, step=global_step)

        epoch_idx = epoch + 1
        if not token_checkpoints:
            _save_epoch_checkpoint(
                state=state,
                config=config,
                epoch_label=f"{epoch_idx:04d}",
                epoch_value=epoch_idx,
                wandb_run_id=wandb.run.id,
            )

    if token_checkpoints:
        wandb.log({"train/tokens_seen": tokens_seen}, step=global_step)
        # The corpus is not an exact multiple of the milestone interval, so the
        # tail after the last milestone would otherwise never be checkpointed.
        if tokens_seen > last_token_checkpoint:
            _save_token_checkpoint(
                state=state,
                config=config,
                nominal_tokens=tokens_seen,
                tokens_seen=tokens_seen,
                epoch_value=float(final_epoch),
                wandb_run_id=wandb.run.id,
            )

    try:
        return _post_training_evals_and_checkpoint(
            state=state,
            model=model,
            config=config,
            rng=rng,
            train_np=train_np,
            val_np=val_np,
            test_np=test_np,
            global_step=global_step,
            final_epoch=final_epoch,
            early_stopped=early_stopped,
            best_test_loss=best_test_loss,
            best_test_step=best_test_step,
        )
    except Exception:
        wandb.finish(exit_code=1)
        raise

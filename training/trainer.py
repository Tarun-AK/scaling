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
from training.loss import cross_entropy_loss
from training.metrics import compute_all_ngram_losses


class TrainState(train_state.TrainState):
    """Flax TrainState with no extra fields."""


def create_train_state(
    model: LSTMLanguageModel, config: Any, rng: jax.Array
) -> TrainState:
    """Create an initialized TrainState."""
    dummy_tokens = jnp.zeros(
        (int(config.batch_size), int(config.seq_len)), dtype=jnp.int32
    )
    params = model.init(rng, dummy_tokens)["params"]
    tx = optax.adam(learning_rate=float(config.learning_rate))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


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


@functools.partial(jax.jit, static_argnames=("bos_token_id",))
def train_step(
    state: TrainState,
    batch: jax.Array,
    rng: jax.Array,
    bos_token_id: int,
) -> Tuple[TrainState, Dict[str, jax.Array], jax.Array]:
    """Perform a single optimization step."""
    inputs = _prepend_bos(batch, bos_token_id)
    observed_next_tokens = batch

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = cross_entropy_loss(logits, observed_next_tokens)
        return loss, logits

    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, {"loss": loss}, rng


@functools.partial(jax.jit, static_argnames=("bos_token_id",))
def eval_step(
    state: TrainState,
    batch: jax.Array,
    bos_token_id: int,
) -> Dict[str, jax.Array]:
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


def _mean_metrics(metrics_list: Iterable[Dict[str, jax.Array]]) -> Dict[str, float]:
    """Average a list of metric dicts into Python floats."""
    totals: Dict[str, float] = {}
    count = 0
    for m in metrics_list:
        count += 1
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + float(jax.device_get(v))
    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


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
    model: LSTMLanguageModel,
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

        init_token = jnp.full((init_carry[0][0].shape[0],), bos_token_id)
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


def _checkpoint_metadata(config: Any, epoch: int | None) -> dict[str, int | str]:
    metadata: dict[str, int | str] = {
        "hidden_dim": int(config.hidden_dim),
        "num_layers": int(config.num_layers),
        "seq_len": int(config.seq_len),
        "vocab_size": int(config.vocab_size),
    }
    if epoch is not None:
        metadata["epoch"] = int(epoch)
    return metadata


def _save_checkpoint(
    *,
    state: TrainState,
    config: Any,
    checkpoint_root: str,
    wandb_run_id: str,
    epoch: int | None,
) -> str:
    _ensure_dir(checkpoint_root)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        os.path.join(checkpoint_root, "ckpt"),
        {
            "params": state.params,
            "config": _config_to_dict(config),
            "wandb_run_id": wandb_run_id,
            "epoch": epoch,
        },
        force=True,
    )
    return checkpoint_root


def _log_checkpoint_artifact(
    *,
    config: Any,
    checkpoint_root: str,
    artifact_name: str,
    wandb_run_id: str,
    epoch: int | None,
) -> None:
    import wandb

    metadata = {"wandb_run_id": wandb_run_id, **_checkpoint_metadata(config, epoch)}
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata,
    )
    artifact.add_dir(checkpoint_root)
    wandb.log_artifact(artifact)


def _post_training_evals_and_checkpoint(
    *,
    state: TrainState,
    model: LSTMLanguageModel,
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

    final_test_loss = _eval_loss(state, test_np, int(config.batch_size), bos_token_id)
    wandb.log({"test/loss": final_test_loss}, step=global_step)

    test_mean = _eval_split(state, test_np, int(config.batch_size), bos_token_id)
    wandb.log({f"test/{k}": v for k, v in test_mean.items()}, step=global_step)

    combined_np = np.concatenate([val_np, test_np], axis=0)
    combined_mean = _eval_split(
        state, combined_np, int(config.batch_size), bos_token_id
    )
    wandb.log({f"combined/{k}": v for k, v in combined_mean.items()})

    if bool(getattr(config, "log_train_ngram_after_training", False)):
        train_mean = _eval_split(state, train_np, int(config.batch_size), bos_token_id)
        wandb.log({f"train_ngram/{k}": v for k, v in train_mean.items()})

    entropy_num_samples = int(getattr(config, "entropy_num_samples", 0))
    entropy_batch_size = int(getattr(config, "entropy_sample_batch_size", 0))
    if entropy_num_samples > 0 and entropy_batch_size > 0:
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
            wandb.log({f"conditional_entropy/{k}": v for k, v in entropy_dict.items()})

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
    model = LSTMLanguageModel(
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        vocab_size=int(config.vocab_size),
    )
    rng = jax.random.PRNGKey(0)
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(model, config, init_rng)
    _wandb_init(config)
    import wandb

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    wandb.config.update({"num_params": num_params})

    # Initial validation before any training
    initial_val_mean = _eval_split(state, val_np, int(config.batch_size), bos_token_id)
    wandb.log({f"val/{k}": v for k, v in initial_val_mean.items()}, step=0)

    eval_every_n_steps = int(getattr(config, "eval_every_n_steps", 0))
    early_stop_patience = int(getattr(config, "early_stop_patience", 0))
    early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 0.0))
    best_test_loss: float | None = None
    best_test_step: int | None = None
    plateau_count = 0

    if eval_every_n_steps > 0:
        best_test_loss = _eval_loss(
            state, test_np, int(config.batch_size), bos_token_id
        )
        best_test_step = 0
        wandb.log({"test/loss": best_test_loss}, step=0)

    wandb.log({"epoch": 0}, step=0)

    global_step = 0
    early_stopped = False
    final_epoch = 0
    batch_size = int(config.batch_size)
    num_train_batches = train_np.shape[0] // batch_size
    num_epochs = int(config.num_epochs)

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
        for batch in progress:
            state, metrics, rng = train_step(state, batch, rng, bos_token_id)
            global_step += 1
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
                    int(config.batch_size),
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
        val_mean = _eval_split(state, val_np, int(config.batch_size), bos_token_id)
        wandb.log({f"val/{k}": v for k, v in val_mean.items()}, step=global_step)

        epoch_idx = epoch + 1
        epoch_checkpoint_root = os.path.join(
            str(config.checkpoint_dir),
            "epochs",
            f"hidden_dim={int(config.hidden_dim)}",
            f"epoch={epoch_idx:04d}",
        )
        _save_checkpoint(
            state=state,
            config=config,
            checkpoint_root=epoch_checkpoint_root,
            wandb_run_id=wandb.run.id,
            epoch=epoch_idx,
        )
        _log_checkpoint_artifact(
            config=config,
            checkpoint_root=epoch_checkpoint_root,
            artifact_name=f"checkpoint-{wandb.run.id}-epoch-{epoch_idx:04d}",
            wandb_run_id=wandb.run.id,
            epoch=epoch_idx,
        )

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

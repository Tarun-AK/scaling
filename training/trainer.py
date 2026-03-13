"""Training loop for LSTM scaling experiments.

Key features:
- JIT-compiled train/eval steps.
- Weights & Biases logging.
- Fixed number of epochs (no early stopping).
- Conditional entropy computed by sampling from p_theta at end of each epoch.
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


@jax.jit
def train_step(
    state: TrainState, batch: jax.Array
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """Perform a single optimization step."""
    inputs = batch[:, :-1]
    observed_next_tokens = batch[:, 1:]

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = cross_entropy_loss(logits, observed_next_tokens)
        return loss, logits

    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, {"loss": loss}


@jax.jit
def eval_step(state: TrainState, batch: jax.Array) -> Dict[str, jax.Array]:
    """Compute all n-gram losses for a batch."""
    return compute_all_ngram_losses(state.apply_fn, state.params, batch)


def _mean_metrics(metrics_list: Iterable[Dict[str, jax.Array]]) -> Dict[str, float]:
    """Average a list of metric dicts into Python floats."""
    stacked: Dict[str, list[jax.Array]] = {}
    for m in metrics_list:
        for k, v in m.items():
            stacked.setdefault(k, []).append(v)
    return {
        k: float(jax.device_get(jnp.mean(jnp.stack(vs)))) for k, vs in stacked.items()
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _wandb_init(config: Any) -> None:
    import wandb

    cfg_dict = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else dict(config)
    )
    wandb.init(
        project=str(getattr(config, "wandb_project", "scaling")),
        entity=getattr(config, "wandb_entity", None),
        group=getattr(config, "wandb_group", None),
        name=getattr(config, "run_name", None),
        config=cfg_dict,
    )


@functools.partial(
    jax.jit, static_argnames=("apply_fn", "seq_len", "batch_size", "vocab_size")
)
def _sample_batch(
    apply_fn,
    params,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    rng: jax.Array,
) -> jax.Array:
    """Sample a single batch of sequences autoregressively using lax.scan."""

    def scan_step(carry, _):
        tokens, rng = carry
        logits = apply_fn({"params": params}, tokens)
        next_logits = logits[:, -1, :]
        rng, step_rng = jax.random.split(rng)
        next_token = jax.random.categorical(step_rng, next_logits)
        # Shift window left, append new token at end
        tokens = jnp.concatenate([tokens[:, 1:], next_token[:, None]], axis=1)
        return (tokens, rng), next_token

    rng, init_rng = jax.random.split(rng)
    # Fixed shape buffer: start with one random token, rest zeros
    init_tokens = jnp.concatenate(
        [
            jax.random.randint(
                init_rng, shape=(batch_size, 1), minval=0, maxval=vocab_size
            ),
            jnp.zeros((batch_size, seq_len - 1), dtype=jnp.int32),
        ],
        axis=1,
    )

    (_, _), sampled = jax.lax.scan(
        scan_step, (init_tokens, rng), None, length=seq_len - 1
    )

    # sampled: (seq_len-1, batch_size) -> (batch_size, seq_len)
    return jnp.concatenate(
        [
            init_tokens[:, :1],
            jnp.transpose(sampled),
        ],
        axis=1,
    )


def sample_sequences(
    apply_fn,
    params,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    vocab_size: int,
    rng: jax.Array,
) -> np.ndarray:
    """Sample sequences autoregressively from p_theta."""
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in tqdm(range(num_batches), desc="Sampling sequences"):
        rng, batch_rng = jax.random.split(rng)
        batch_samples = _sample_batch(
            apply_fn=apply_fn,
            params=params,
            seq_len=seq_len,
            batch_size=batch_size,
            vocab_size=vocab_size,
            rng=batch_rng,
        )
        all_samples.append(np.array(batch_samples))

    return np.concatenate(all_samples, axis=0)[:num_samples]


def compute_conditional_entropy(
    apply_fn,
    params,
    samples: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    """Compute per-position conditional entropy H_n(p_theta) from sampled sequences."""
    all_metrics = []
    num_batches = len(samples) // batch_size

    for i in tqdm(range(num_batches), desc="Computing entropy"):
        batch = jnp.array(samples[i * batch_size : (i + 1) * batch_size])
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)

        per_position = -jnp.mean(token_logp, axis=0)
        all_metrics.append(np.array(per_position))

    mean_per_position = np.mean(np.stack(all_metrics), axis=0)
    return {
        f"entropy_{n + 1}": float(mean_per_position[n])
        for n in range(len(mean_per_position))
    }


def train_and_evaluate(config: Any) -> Dict[str, float]:
    """Run training loop, final test evaluation, and checkpointing."""
    _ensure_dir(str(config.checkpoint_dir))
    _ensure_dir(str(config.results_dir))

    # Data
    train_np, val_np, test_np = load_splits_as_arrays(
        dataset_name=str(config.dataset_name),
        dataset_config=str(config.dataset_config),
        seq_len=int(config.seq_len),
        vocab_size=int(config.vocab_size),
    )

    # Model/state
    model = LSTMLanguageModel(
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        vocab_size=int(config.vocab_size),
    )
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, config, rng)
    _wandb_init(config)
    import wandb

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    wandb.config.update({"num_params": num_params})

    # Initial validation before any training
    initial_val_metrics = [
        eval_step(state, batch)
        for batch in batch_iterator(
            val_np,
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=0,
            drop_last=True,
        )
    ]
    initial_val_mean = _mean_metrics(initial_val_metrics) if initial_val_metrics else {}
    wandb.log({f"val/{k}": v for k, v in initial_val_mean.items()}, step=0)
    wandb.log({"epoch": 0}, step=0)

    global_step = 0
    entropy_every = int(getattr(config, "entropy_every_n_epochs", 1))
    entropy_num_samples = int(getattr(config, "entropy_num_samples", 10000))
    entropy_sample_batch_size = int(getattr(config, "entropy_sample_batch_size", 512))

    for epoch in range(int(config.num_epochs)):
        # Training
        for batch in batch_iterator(
            train_np,
            batch_size=int(config.batch_size),
            shuffle=True,
            seed=epoch,
            drop_last=True,
        ):
            state, metrics = train_step(state, batch)
            global_step += 1
            if global_step % int(config.log_every_n_steps) == 0:
                wandb.log(
                    {
                        f"train/{k}": float(jax.device_get(v))
                        for k, v in metrics.items()
                    },
                    step=global_step,
                )

        # Validation at end of each epoch
        val_metrics = [
            eval_step(state, batch)
            for batch in batch_iterator(
                val_np,
                batch_size=int(config.batch_size),
                shuffle=False,
                seed=0,
                drop_last=True,
            )
        ]
        val_mean = _mean_metrics(val_metrics) if val_metrics else {}
        wandb.log({f"val/{k}": v for k, v in val_mean.items()}, step=global_step)

        # Conditional entropy by sampling from p_theta
        if (epoch + 1) % entropy_every == 0:
            rng, entropy_rng = jax.random.split(rng)
            samples = sample_sequences(
                apply_fn=state.apply_fn,
                params=state.params,
                seq_len=int(config.seq_len),
                num_samples=entropy_num_samples,
                batch_size=entropy_sample_batch_size,
                vocab_size=int(config.vocab_size),
                rng=entropy_rng,
            )
            entropy_dict = compute_conditional_entropy(
                apply_fn=state.apply_fn,
                params=state.params,
                samples=samples,
                batch_size=entropy_sample_batch_size,
            )
            wandb.log(
                {f"conditional_entropy/{k}": v for k, v in entropy_dict.items()},
                step=global_step,
            )

    # Test evaluation
    test_metrics = [
        eval_step(state, batch)
        for batch in batch_iterator(
            test_np,
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=0,
            drop_last=True,
        )
    ]
    test_mean = _mean_metrics(test_metrics) if test_metrics else {}
    wandb.log({f"test/{k}": v for k, v in test_mean.items()}, step=global_step)

    # Checkpoint
    ckpt_dir = os.path.join(
        str(config.checkpoint_dir), f"hidden_dim={int(config.hidden_dim)}"
    )
    _ensure_dir(ckpt_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        os.path.join(ckpt_dir, "ckpt"),
        {
            "params": state.params,
            "config": (
                OmegaConf.to_container(config, resolve=True)
                if OmegaConf.is_config(config)
                else dict(config)
            ),
        },
        force=True,
    )

    wandb.finish()
    return test_mean

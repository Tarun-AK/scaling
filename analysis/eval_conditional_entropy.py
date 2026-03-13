"""Evaluate conditional entropies by sampling from p_theta.

For each checkpoint, generates num_samples sequences by sampling autoregressively
from the model, then evaluates -log p_theta(x_i | x_{<i}) on those samples.
This gives the model's own conditional entropy H(p_theta), as opposed to the
cross-entropy evaluated on the data distribution.

Samples are saved as W&B artifacts for reproducibility.

Usage:
    python analysis/eval_conditional_entropy.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import wandb
from data.dataset import load_splits_as_arrays
from models.lstm import LSTMLanguageModel
from training.trainer import TrainState, _mean_metrics, create_train_state

CHECKPOINT_DIR = "checkpoints"
WANDB_PROJECT = "tarunadvaith-/scaling"
NUM_SAMPLES = 100_000
SAMPLE_BATCH_SIZE = 512


def load_checkpoint(ckpt_path: str, state: TrainState) -> TrainState:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    return state.replace(params=restored["params"])


def sample_sequences(
    apply_fn,
    params,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    vocab_size: int,
    rng: jax.Array,
) -> np.ndarray:
    """Sample sequences autoregressively from p_theta.

    Args:
        apply_fn: Flax apply function.
        params: Model parameters.
        seq_len: Length of sequences to generate.
        num_samples: Total number of sequences to generate.
        batch_size: Number of sequences to generate in parallel.
        vocab_size: Vocabulary size for sampling.
        rng: JAX PRNG key.

    Returns:
        Array of shape (num_samples, seq_len) of sampled token ids.
    """
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        rng, sample_rng, init_rng = jax.random.split(rng, 3)
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # Start with a random initial token
        tokens = jax.random.randint(
            init_rng, shape=(current_batch_size, 1), minval=0, maxval=vocab_size
        )

        for t in range(seq_len - 1):
            logits = apply_fn({"params": params}, tokens)  # (batch, t+1, vocab)
            next_logits = logits[:, -1, :]  # (batch, vocab)
            rng, step_rng = jax.random.split(rng)
            next_token = jax.random.categorical(step_rng, next_logits)  # (batch,)
            tokens = jnp.concatenate(
                [tokens, next_token[:, None]], axis=1
            )  # (batch, t+2)

        all_samples.append(np.array(tokens))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Sampled {(batch_idx + 1) * batch_size} / {num_samples} sequences")

    return np.concatenate(all_samples, axis=0)[:num_samples]


def compute_conditional_entropy_from_samples(
    apply_fn,
    params,
    samples: np.ndarray,
    batch_size: int,
) -> dict:
    """Compute per-position conditional entropy H_n(p_theta) from sampled sequences.

    Args:
        apply_fn: Flax apply function.
        params: Model parameters.
        samples: Sampled sequences, shape (num_samples, seq_len).
        batch_size: Batch size for forward passes.

    Returns:
        Dict mapping "entropy_n" to scalar conditional entropy at position n.
    """
    all_metrics = []
    num_batches = len(samples) // batch_size

    for i in range(num_batches):
        batch = jnp.array(samples[i * batch_size : (i + 1) * batch_size])
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(
            -1
        )  # (batch, seq_len-1)

        per_position = -jnp.mean(token_logp, axis=0)  # (seq_len-1,)
        all_metrics.append(np.array(per_position))

    mean_per_position = np.mean(np.stack(all_metrics), axis=0)  # (seq_len-1,)
    return {
        f"entropy_{n+1}": float(mean_per_position[n])
        for n in range(len(mean_per_position))
    }


def main() -> None:
    api = wandb.Api()

    hidden_dim_dirs = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("hidden_dim=")],
        key=lambda x: int(x.split("=")[1]),
    )

    if not hidden_dim_dirs:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return

    for hd_dir in hidden_dim_dirs:
        hidden_dim = int(hd_dir.split("=")[1])
        ckpt_path = os.path.abspath(os.path.join(CHECKPOINT_DIR, hd_dir, "ckpt"))
        print(f"\nProcessing hidden_dim={hidden_dim}...")

        # Find matching W&B run
        runs = [
            r
            for r in api.runs(WANDB_PROJECT)
            if r.config.get("hidden_dim") == hidden_dim and r.state == "finished"
        ]
        if not runs:
            print(f"  No finished W&B run found for hidden_dim={hidden_dim}, skipping.")
            continue
        run = runs[0]
        cfg = run.config

        # Rebuild model and load checkpoint
        model = LSTMLanguageModel(
            hidden_dim=hidden_dim,
            num_layers=int(cfg["num_layers"]),
            vocab_size=int(cfg["vocab_size"]),
        )

        class _Cfg:
            pass

        state_cfg = _Cfg()
        for k, v in cfg.items():
            setattr(state_cfg, k, v)

        rng = jax.random.PRNGKey(42)
        state = create_train_state(model, state_cfg, rng)
        state = load_checkpoint(ckpt_path, state)

        # Sample sequences
        print(f"  Sampling {NUM_SAMPLES} sequences...")
        rng, sample_rng = jax.random.split(rng)
        samples = sample_sequences(
            apply_fn=state.apply_fn,
            params=state.params,
            seq_len=int(cfg["seq_len"]),
            num_samples=NUM_SAMPLES,
            batch_size=SAMPLE_BATCH_SIZE,
            vocab_size=int(cfg["vocab_size"]),
            rng=sample_rng,
        )

        # Save samples and upload as W&B artifact
        print(f"  Uploading samples as W&B artifact...")
        wandb.init(
            project=WANDB_PROJECT.split("/")[-1],
            entity=WANDB_PROJECT.split("/")[0],
            id=run.id,
            resume="must",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = os.path.join(tmpdir, f"samples_hd{hidden_dim}.npy")
            np.save(samples_path, samples)

            artifact = wandb.Artifact(
                name=f"samples_hd{hidden_dim}",
                type="samples",
                description=f"Autoregressive samples from p_theta, hidden_dim={hidden_dim}",
            )
            artifact.add_file(samples_path)
            wandb.log_artifact(artifact)

        print(f"  Computing conditional entropies...")
        entropy_dict = compute_conditional_entropy_from_samples(
            apply_fn=state.apply_fn,
            params=state.params,
            samples=samples,
            batch_size=SAMPLE_BATCH_SIZE,
        )

        wandb.log({f"conditional_entropy/{k}": v for k, v in entropy_dict.items()})
        wandb.finish()

        print(f"  Done. entropy_1={entropy_dict.get('entropy_1', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

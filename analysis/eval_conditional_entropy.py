"""Evaluate conditional entropies by sampling from p_theta.

For each checkpoint, generates num_samples sequences by sampling autoregressively
from the model, then evaluates -log p_theta(x_i | x_{<i}) on those samples.
This gives the model's own conditional entropy H(p_theta), as opposed to the
cross-entropy evaluated on the data distribution.

Usage:
    python analysis/eval_conditional_entropy.py --group <group>
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import wandb
from tqdm import tqdm
from models.lstm import LSTMLanguageModel
from training.trainer import TrainState, create_train_state

WANDB_PROJECT = "tarunadvaith-/scaling"
NUM_SAMPLES = 1_000_000
SAMPLE_BATCH_SIZE = 512


def _normalize_params(params: dict, num_layers: int) -> dict:
    if any(k.startswith("rnn_") for k in params):
        return params

    lstm_keys = [k for k in params if k.startswith("LSTMCell_")]
    if not lstm_keys:
        raise RuntimeError("Unrecognized checkpoint parameter structure")

    normalized = dict(params)
    for layer_idx in range(num_layers):
        cell_key = f"LSTMCell_{layer_idx}"
        if cell_key not in params:
            raise RuntimeError(
                f"Missing {cell_key} in checkpoint params for num_layers={num_layers}"
            )
        normalized.setdefault(f"rnn_{layer_idx}", {"cell": params[cell_key]})
    return normalized


def load_checkpoint(
    ckpt_path: str, state: TrainState, num_layers: int
) -> tuple[TrainState, dict]:
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    params = _normalize_params(restored["params"], num_layers)
    return state.replace(params=params), restored


def _wandb_entity_project() -> tuple[str, str]:
    entity, project = WANDB_PROJECT.split("/")
    return entity, project


def _download_checkpoint_artifact(run_id: str, api: wandb.Api) -> str:
    entity, project = _wandb_entity_project()
    artifact_name = f"{entity}/{project}/checkpoint-{run_id}:latest"
    artifact = api.artifact(artifact_name)
    tmpdir = tempfile.mkdtemp(prefix="ckpt_artifact_")
    artifact_dir = artifact.download(root=tmpdir)
    return os.path.join(artifact_dir, "ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional entropy")
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="W&B group containing the runs",
    )
    return parser.parse_args()


def sample_sequences_with_logps(
    model: LSTMLanguageModel,
    params: dict,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    bos_token_id: int,
    rng: jax.Array,
) -> np.ndarray:
    """Sample sequences and return per-position log probabilities.

    Returns:
        logps: (num_samples, seq_len) log p(x_t | x_{<t})
    """

    def _sample_batch(
        batch_rng: jax.Array,
        init_carry: tuple,
        seq_len: int,
        bos_token_id: int,
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
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_logp = jnp.take_along_axis(
                log_probs, next_token[:, None], axis=-1
            ).squeeze(-1)
            new_state = {"carry": lstm_carry, "token": next_token, "rng": rng}
            return new_state, token_logp

        init_token = jnp.full((init_carry[0][0].shape[0],), bos_token_id)
        carry = {"carry": init_carry, "token": init_token, "rng": batch_rng}
        _, logps = jax.lax.scan(scan_step, carry, jnp.arange(seq_len))
        return jnp.transpose(logps, (1, 0))

    sample_batch_jit = jax.jit(
        _sample_batch, static_argnames=("seq_len", "bos_token_id")
    )

    all_logps = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Sampling batches"):
        rng, batch_rng = jax.random.split(rng)
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        init_carry = model.init_carry(current_batch_size)
        logps = sample_batch_jit(batch_rng, init_carry, seq_len, bos_token_id)
        all_logps.append(np.array(logps))

    logps = np.concatenate(all_logps, axis=0)[:num_samples]
    return logps


def compute_conditional_entropy_from_samples(
    apply_fn,
    params,
    samples: np.ndarray,
    batch_size: int,
    bos_token_id: int,
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
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Computing entropy"):
        start = i * batch_size
        end = min(start + batch_size, len(samples))
        batch = jnp.array(samples[start:end])
        bos = jnp.full((batch.shape[0], 1), bos_token_id, dtype=batch.dtype)
        inputs = jnp.concatenate([bos, batch[:, :-1]], axis=1)
        targets = batch

        logits = apply_fn({"params": params}, inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)  # (batch, seq_len-1)

        per_position = -jnp.mean(token_logp, axis=0)
        all_metrics.append(np.array(per_position))

    mean_per_position = np.mean(np.stack(all_metrics), axis=0)
    return {
        f"entropy_{n + 1}": float(mean_per_position[n])
        for n in range(len(mean_per_position))
    }


def main() -> None:
    args = parse_args()
    wandb_group = args.group
    api = wandb.Api()

    runs = list(
        api.runs(WANDB_PROJECT, filters={"group": wandb_group, "state": "finished"})
    )

    if not runs:
        raise RuntimeError(f"No finished runs found in group '{wandb_group}'")

    runs = [r for r in runs if (r.config or {}).get("hidden_dim") is not None]
    if not runs:
        raise RuntimeError(
            f"No finished runs with config.hidden_dim found in group '{wandb_group}'"
        )

    for run in sorted(
        runs,
        key=lambda r: int((r.config or {}).get("hidden_dim", 0)),
        reverse=True,
    ):
        cfg = run.config or {}
        hidden_dim = int(cfg["hidden_dim"])

        ckpt_path = _download_checkpoint_artifact(run.id, api)

        print(f"\nProcessing hidden_dim={hidden_dim}, run={run.name}...")

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
        state, restored = load_checkpoint(ckpt_path, state, int(cfg["num_layers"]))
        ckpt_run_id = restored.get("wandb_run_id")
        if ckpt_run_id is None:
            raise RuntimeError(
                f"Checkpoint at {ckpt_path} is missing wandb_run_id; "
                "cannot verify run alignment."
            )
        if ckpt_run_id != run.id:
            raise RuntimeError(
                "Checkpoint/run mismatch for hidden_dim="
                f"{hidden_dim}: ckpt_run_id={ckpt_run_id}, run.id={run.id}"
            )

        # Sample sequences
        print(f"  Sampling {NUM_SAMPLES} sequences...")
        rng, sample_rng = jax.random.split(rng)
        bos_token_id = int(cfg.get("bos_token_id", 0))
        logps = sample_sequences_with_logps(
            model=model,
            params=state.params,
            seq_len=int(cfg["seq_len"]),
            num_samples=NUM_SAMPLES,
            batch_size=SAMPLE_BATCH_SIZE,
            bos_token_id=bos_token_id,
            rng=sample_rng,
        )

        wandb.init(
            project=WANDB_PROJECT.split("/")[-1],
            entity=WANDB_PROJECT.split("/")[0],
            id=run.id,
            resume="must",
        )

        print(f"  Computing conditional entropies...")
        per_position = -np.mean(logps, axis=0)
        entropy_dict = {
            f"entropy_{n + 1}": float(per_position[n]) for n in range(len(per_position))
        }

        wandb.log({f"conditional_entropy/{k}": v for k, v in entropy_dict.items()})
        wandb.finish()

        print(f"  Done. entropy_1={entropy_dict.get('entropy_1', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

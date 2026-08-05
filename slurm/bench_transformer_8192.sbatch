#!/bin/bash
#SBATCH --job-name=bench-tf-8192
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=slurm_logs/%x/%j.out
#SBATCH --error=slurm_logs/%x/%j.err
#SBATCH --gpus=h100:1
#SBATCH --account=def-rgmelko

# Calibration only -- measures real train_step wall time for candidate
# transformer shapes at seq_len=8192 so the 20h target can be solved from
# measurement instead of an assumed TFLOPs number.
#
# Uses random tokens, so it does NOT need the tokenized cache and can run
# while owt_full_tokenize_8192.sbatch is still going. Short walltime keeps it
# high in the backfill queue.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "${REPO_ROOT}/slurm_logs/${SLURM_JOB_NAME}"
source "${REPO_ROOT}/.venv/bin/activate"

python - <<'PY'
import time, jax, jax.numpy as jnp, numpy as np
from training.trainer import create_train_state, train_step
from models.transformer import TransformerLanguageModel

V, S, B = 10000, 8192, 64
TOTAL_TOKENS = 10_109_000_000
STEPS = TOTAL_TOKENS // (B * S)

class Cfg:
    batch_size = B
    learning_rate = 1e-3
    lr_schedule = "constant"
    warmup_steps = 0
    min_learning_rate = 0.0
    num_epochs = 1

print(f"device: {jax.devices()[0]}  steps for full corpus: {STEPS:,}", flush=True)
print(f"{'L':>3}{'d':>6}{'h':>4} {'params':>13} {'s/step':>8} {'proj_hours':>11}", flush=True)

for L, d, h in [(8,512,8), (10,512,8), (8,768,8), (10,768,8), (12,768,8)]:
    try:
        model = TransformerLanguageModel(
            hidden_dim=d, num_layers=L, vocab_size=V, num_heads=h, num_kv_heads=h,
            ffn_mult=8/3, rope_theta=1e6, layer_norm_epsilon=1e-6,
            tie_word_embeddings=True)
        cfg = Cfg()
        cfg.hidden_dim, cfg.num_layers, cfg.seq_len, cfg.vocab_size = d, L, S, V
        state = create_train_state(model, jax.random.PRNGKey(0), cfg)
        n = sum(x.size for x in jax.tree_util.tree_leaves(state.params))

        batch = jnp.asarray(np.random.randint(0, V, size=(B, S), dtype=np.int32))
        rng = jax.random.PRNGKey(1)
        # grad_accum_steps=64 -> microbatch 1, forced by materialized attention
        state, m, rng = train_step(state, batch, rng, 0, 64)
        jax.block_until_ready(m["loss"])                      # compile

        t0 = time.time()
        for _ in range(3):
            state, m, rng = train_step(state, batch, rng, 0, 64)
        jax.block_until_ready(m["loss"])
        dt = (time.time() - t0) / 3
        print(f"{L:>3}{d:>6}{h:>4} {n:>13,} {dt:>8.2f} {dt*STEPS/3600:>10.1f}h", flush=True)
        del state
    except Exception as e:
        print(f"{L:>3}{d:>6}{h:>4} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
PY

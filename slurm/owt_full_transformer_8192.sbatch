#!/bin/bash
#SBATCH --job-name=owt-full-tf-8192
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x/%j.out
#SBATCH --error=slurm_logs/%x/%j.err
#SBATCH --gpus=h100:1
#SBATCH --account=def-rgmelko

# ~111M-parameter transformer at seq_len=8192 on full OpenWebText, matched to
# the mamba sweep (same corpus, seq_len, effective batch, token-milestone
# checkpoints) so the two are directly comparable.
#
# Shape: L=8, d=1024, 8 heads (head_dim 128) -> 110,904,320 params. Chosen over
# the 13x768 alternative because it carries MORE parameters at the same cost:
# params grow as L*d^2 while attention grows as L*s*d, so width is the cheaper
# way to buy capacity at long seq_len. Fewer layers also cuts activation memory.
#
# transformer_attention_impl=cudnn selects the fused flash kernel. Without it
# the seq x seq score matrix is materialized (~2.1GB per layer per sequence at
# fp32) and training becomes memory-bandwidth bound -- ~21h instead of ~11h.
# cuDNN accepts only fp16/bf16, so models/transformer.py runs the attention
# core in bf16 and returns to fp32; projections stay fp32. If cuDNN cannot
# serve this shape it RAISES rather than silently falling back, so a wrong
# assumption fails fast instead of quietly costing 10 hours.
#
# batch_size is the EFFECTIVE batch (see trainer.py:744); grad_accum_steps=8
# gives a microbatch of 8, which fits comfortably once flash attention removes
# the ~28GB of retained softmax weights.

set -euo pipefail
module load python

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${REPO_ROOT}/.venv"

mkdir -p "${REPO_ROOT}/slurm_logs/${SLURM_JOB_NAME}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

CORPUS_DIR="${CORPUS_DIR:-/project/def-rgmelko/tarunak/owt_full}"
CACHE_DIR="${CACHE_DIR:-/project/def-rgmelko/tarunak/owt_full_cache}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/openwebtext/tokenizer/tokenizer.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/project/def-rgmelko/tarunak/owt_full_checkpoints}"

python -m experiments.scaling_hidden_dim \
  --config-name openwebtext_full \
  model_type=transformer \
  hidden_dim=1024 \
  num_layers=8 \
  transformer_num_heads=8 \
  transformer_num_kv_heads=8 \
  transformer_attention_impl=cudnn \
  seq_len=8192 \
  batch_size=64 \
  grad_accum_steps=8 \
  eval_batch_size=4 \
  require_cached_data=true \
  dataset_path="${CORPUS_DIR}" \
  cache_dir="${CACHE_DIR}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  checkpoint_dir="${CHECKPOINT_DIR}"

#!/bin/bash
#SBATCH --job-name=owt-full-mamba-8192
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x/%A_%a.out
#SBATCH --error=slurm_logs/%x/%A_%a.err
#SBATCH --gpus=h100:1
#SBATCH --account=def-rgmelko

# Mamba hidden_dim sweep at seq_len=8192 on the FULL OpenWebText corpus
# (~10.3B tokens), with token-milestone checkpointing.
#
# Run only after slurm/owt_full_tokenize_8192.sbatch has produced the cache.
#
# batch_size is the EFFECTIVE batch here, not a microbatch: training/trainer.py
# hands config.batch_size straight to batch_iterator, and _train_step_jit then
# reshapes that batch into grad_accum_steps microbatches and averages the grads
# (hence the batch_size % grad_accum_steps check at trainer.py:744). So
# batch_size=64 is effective-64 regardless of grad_accum_steps; accumulation
# only trades step time for peak memory. grad_accum_steps=8 -> microbatch of 8.

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

HIDDEN_DIMS=(128 160 192 256 320 384 512)
HIDDEN_DIM="${HIDDEN_DIMS[$SLURM_ARRAY_TASK_ID]}"

CORPUS_DIR="${CORPUS_DIR:-/project/def-rgmelko/tarunak/owt_full}"
CACHE_DIR="${CACHE_DIR:-/project/def-rgmelko/tarunak/owt_full_cache}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/openwebtext/tokenizer/tokenizer.json}"
# Off /home deliberately: /home is at 46/50 GiB. Each run writes 8 checkpoints
# and wandb also stages every artifact locally before upload, so keeping these
# in the repo risks running out of quota hours into a GPU job.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/project/def-rgmelko/tarunak/owt_full_checkpoints}"

python -m experiments.scaling_hidden_dim \
  --config-name openwebtext_full \
  model_type=mamba \
  hidden_dim="${HIDDEN_DIM}" \
  seq_len=8192 \
  batch_size=64 \
  grad_accum_steps=8 \
  eval_batch_size=4 \
  require_cached_data=true \
  dataset_path="${CORPUS_DIR}" \
  cache_dir="${CACHE_DIR}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  checkpoint_dir="${CHECKPOINT_DIR}"

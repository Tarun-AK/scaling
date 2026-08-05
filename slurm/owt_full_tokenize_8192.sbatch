#!/bin/bash
#SBATCH --job-name=owt-full-tok-8192
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%x/%j.out
#SBATCH --error=slurm_logs/%x/%j.err
#SBATCH --account=def-rgmelko

# Tokenize the full-OpenWebText corpus into a seq_len=8192 cache.
#
# CPU-only: no GPU and no internet needed (the corpus is already on disk, and
# the tokenizer is a local file). Run this AFTER preprocess_openwebtext_full.py
# has finished writing the corpus.
#
# Cost note: data/dataset.py:_tokenize_to_memmap makes TWO passes over the
# corpus -- _count_tokens tokenizes everything to size the memmap, then the
# fill pass tokenizes it again. Budget ~2x the naive estimate. encode_batch is
# Rust-multithreaded, so cpus-per-task is the main throughput lever.

set -euo pipefail
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

# Guard: manifest.json is written only after preprocess_openwebtext_full.py has
# closed every split writer, so its presence means the corpus is complete. The
# split .txt files reach full size BEFORE the build finishes, so checking for
# those instead would happily tokenize a truncated corpus and produce a smaller
# cache with no error at all.
if [[ ! -f "${CORPUS_DIR}/manifest.json" ]]; then
  echo "Corpus incomplete: ${CORPUS_DIR}/manifest.json missing" >&2
  exit 1
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

python data/preprocess_wikitext.py \
  --dataset-name openwebtext_full \
  --dataset-config none \
  --seq-len 8192 \
  --vocab-size 10000 \
  --cache-dir "${CACHE_DIR}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --dataset-path "${CORPUS_DIR}" \
  --tokenize-batch-size 1000

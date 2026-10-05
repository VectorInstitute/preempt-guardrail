#!/bin/bash
# Fine-tune English DistilBERT for PII spans.
# Fill in --partition and --account for your cluster, then:
#   mkdir -p logs
#   sbatch training/train.sbatch
#SBATCH --job-name=pii-distilbert
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"

OUT="${SCRATCH:-/tmp}/preempt-distilbert/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

uv run --group train python -m training.prepare --output-dir "$OUT/data"
uv run --group train python -m training.train --data-dir "$OUT/data" --output-dir "$OUT"

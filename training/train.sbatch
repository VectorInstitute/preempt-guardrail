#!/bin/bash
# Fine-tune English DistilBERT for PII spans.
#   mkdir -p logs
#   sbatch --export=ALL training/train.sbatch
#SBATCH --job-name=pii-distilbert
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Running on $(hostname)"
date

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HOME=$HOME/.cache/huggingface

OUT=$HOME/preempt-distilbert/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "Writing run data to $OUT"

uv run --group train python -m training.prepare --output-dir "$OUT/data"
uv run --group train python -m training.train --data-dir "$OUT/data" --output-dir "$OUT"

#!/bin/bash
# Write AGE and MONEY misses from the 407111 checkpoint.
#   mkdir -p logs
#   sbatch --export=ALL training/misses.sbatch
#SBATCH --job-name=pii-misses
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
SLURM_JOB_ID=${SLURM_JOB_ID:-}

echo "Running on $(hostname)"
date

export TOKENIZERS_PARALLELISM=false
export HF_HOME=$HOME/.cache/huggingface

OUT=$HOME/preempt-distilbert/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "Writing misses to $OUT"

uv run --group train python -m training.misses \
  --checkpoint "$HOME/preempt-distilbert/$SLURM_JOB_ID/checkpoint" \
  --test "$HOME/preempt-distilbert/$SLURM_JOB_ID/data/test.jsonl" \
  --output-dir "$OUT"

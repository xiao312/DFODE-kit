#!/usr/bin/env bash
#SBATCH --job-name=dfode-h2-cvode
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load compiler/dtk/26.04

export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
if [[ -n "${DFODE_TORCH_SITE:-}" ]]; then
  export PYTHONPATH="$DFODE_TORCH_SITE${PYTHONPATH:+:$PYTHONPATH}"
fi
: "${DFODE_ROOT:?set DFODE_ROOT}"
: "${TRAIN_SOURCE:?set TRAIN_SOURCE}"
: "${VALIDATION_SOURCE:?set VALIDATION_SOURCE}"
: "${MECHANISM:?set MECHANISM}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${PYTHON:?set PYTHON to the DCU PyTorch interpreter}"

mkdir -p logs "$OUTPUT_ROOT"
cd "$DFODE_ROOT"

"$PYTHON" scripts/train_positive_interval_distributed.py \
  --train-source "$TRAIN_SOURCE" \
  --validation-source "$VALIDATION_SOURCE" \
  --mechanism "$MECHANISM" \
  --output "$OUTPUT_ROOT/best.pt" \
  --final-output "$OUTPUT_ROOT/final.pt" \
  --resume-output "$OUTPUT_ROOT/resume.pt" \
  --epochs "${EPOCHS:-2000}" \
  --batch-size-per-device "${BATCH_SIZE:-4096}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --minimum-learning-rate "${MINIMUM_LEARNING_RATE:-1e-5}" \
  --hidden-dim "${HIDDEN_DIM:-128}" \
  --latent-dim "${LATENT_DIM:-32}" \
  --transform-alpha 0.1 \
  --tp-loss-weight 1.0 \
  --state-loss-weight 1.0 \
  --delta-loss-weight 0.1 \
  --relative-delta-loss-weight 0.1 \
  --direction-loss-weight 0.25 \
  --source-loss-weight 0.05 \
  --source-loss-scale 1000 \
  --extent-scale 1e-4 \
  --seed 260816 \
  --log-every 10 \
  --checkpoint-every 25 \
  --device cuda

"$PYTHON" scripts/build_ood_envelope.py \
  --checkpoint "$OUTPUT_ROOT/best.pt" \
  --dataset "$TRAIN_SOURCE" \
  --output "$OUTPUT_ROOT/ood-envelope.json"

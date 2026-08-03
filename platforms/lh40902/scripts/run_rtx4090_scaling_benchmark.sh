#!/bin/bash
set -euo pipefail

: "${DFODE_SOURCE_ROOT:?Set DFODE_SOURCE_ROOT}"
: "${DFODE_TRAIN_SOURCE:?Set DFODE_TRAIN_SOURCE}"
: "${DFODE_VALIDATION_SOURCE:?Set DFODE_VALIDATION_SOURCE}"
: "${DFODE_MECHANISM:?Set DFODE_MECHANISM}"
: "${DFODE_BENCHMARK_ROOT:?Set DFODE_BENCHMARK_ROOT}"

PYTHON="${DFODE_PYTHON:-python}"
EPOCHS="${DFODE_EPOCHS:-20}"
SEED="${DFODE_SEED:-260624}"
mkdir -p "${DFODE_BENCHMARK_ROOT}"
export PYTHONPATH="${DFODE_SOURCE_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for spec in 1:4096 2:2048 4:1024 6:682; do
    IFS=: read -r gpu_count local_batch <<< "${spec}"
    run_dir="${DFODE_BENCHMARK_ROOT}/gpu-${gpu_count}"
    mkdir -p "${run_dir}"
    visible_devices="$(seq -s, 0 $((gpu_count - 1)))"
    echo "gpu_count=${gpu_count} local_batch=${local_batch}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        "${PYTHON}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${gpu_count}" \
        "${DFODE_SOURCE_ROOT}/scripts/train_positive_interval_distributed.py" \
        --train-source "${DFODE_TRAIN_SOURCE}" \
        --validation-source "${DFODE_VALIDATION_SOURCE}" \
        --mechanism "${DFODE_MECHANISM}" \
        --output "${run_dir}/checkpoint.pt" \
        --final-output "${run_dir}/final.pt" \
        --resume-output "${run_dir}/last.pt" \
        --epochs "${EPOCHS}" \
        --batch-size-per-device "${local_batch}" \
        --learning-rate "${DFODE_LEARNING_RATE:-1e-3}" \
        --minimum-learning-rate "${DFODE_MINIMUM_LEARNING_RATE:-1e-5}" \
        --hidden-dim "${DFODE_HIDDEN_DIM:-128}" \
        --latent-dim "${DFODE_LATENT_DIM:-32}" \
        --transform-alpha "${DFODE_TRANSFORM_ALPHA:-0.1}" \
        --delta-loss-weight "${DFODE_DELTA_LOSS_WEIGHT:-0.1}" \
        --extent-scale "${DFODE_EXTENT_SCALE:-1e-4}" \
        --seed "${SEED}" \
        --log-every "${DFODE_LOG_EVERY:-10}" \
        --checkpoint-every "${DFODE_CHECKPOINT_EVERY:-10}" \
        2>&1 | tee "${run_dir}/train.log"
done

"${PYTHON}" "${DFODE_SOURCE_ROOT}/scripts/summarize_gpu_scaling.py" \
    --benchmark-root "${DFODE_BENCHMARK_ROOT}" \
    --output "${DFODE_BENCHMARK_ROOT}/scaling-report.json"

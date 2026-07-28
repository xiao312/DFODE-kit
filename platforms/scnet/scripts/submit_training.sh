#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLURM_SCRIPT="${REPO_ROOT}/platforms/scnet/slurm/run_training.slurm"

: "${DFODE_DCU_PYTHON:?DFODE_DCU_PYTHON is required}"
: "${DFODE_WHEEL_DIR:?DFODE_WHEEL_DIR is required}"
: "${DFODE_TRAIN_DATA:?DFODE_TRAIN_DATA is required}"
: "${DFODE_MECHANISM:?DFODE_MECHANISM is required}"

RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_dfode_train_dcu}"
RUN_DIR="${DFODE_RUN_ROOT:-${REPO_ROOT}/.work/runs}/${RUN_ID}"
LOG_DIR="${DFODE_LOG_ROOT:-${REPO_ROOT}/.work/logs}/${RUN_ID}"
OUTPUT_PATH="${DFODE_OUTPUT_PATH:-${RUN_DIR}/checkpoint.pt}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

job_id="$(
    sbatch \
        --parsable \
        --output="${LOG_DIR}/%x-%j.out" \
        --error="${LOG_DIR}/%x-%j.err" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",RUN_ID="${RUN_ID}",RUN_DIR="${RUN_DIR}",DFODE_OUTPUT_PATH="${OUTPUT_PATH}" \
        "${SLURM_SCRIPT}"
)"

printf 'Submitted %s\n' "${job_id}"
printf 'Run directory: %s\n' "${RUN_DIR}"
printf 'Checkpoint: %s\n' "${OUTPUT_PATH}"
printf 'Log directory: %s\n' "${LOG_DIR}"

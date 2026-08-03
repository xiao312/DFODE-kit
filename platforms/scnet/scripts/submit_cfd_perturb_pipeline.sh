#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_JOB="${REPO_ROOT}/platforms/scnet/slurm/run_cfd_perturb_data.slurm"
TRAIN_JOB="${REPO_ROOT}/platforms/scnet/slurm/run_positive_interval.slurm"
GATE_JOB="${REPO_ROOT}/platforms/scnet/slurm/run_cfd_perturb_gate.slurm"

: "${PLUGS_ROOT:?PLUGS_ROOT is required}"
: "${DFODE_DCU_PYTHON:?DFODE_DCU_PYTHON is required}"
: "${CASE_FILE:?CASE_FILE is required}"
: "${DATA_FILE:?DATA_FILE is required}"
: "${SPECIES_FILE:?SPECIES_FILE is required}"
: "${SNAPSHOT_CONFIG:?SNAPSHOT_CONFIG is required}"
: "${DFODE_MECHANISM:?DFODE_MECHANISM is required}"

RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_cfd_perturb}"
PIPELINE_ROOT="${DFODE_PIPELINE_ROOT:-${PLUGS_ROOT}/.work/pipelines}"
PIPELINE_DIR="${PIPELINE_ROOT}/${RUN_ID}"
LOG_DIR="${DFODE_LOG_ROOT:-${PLUGS_ROOT}/.work/logs/cfd-perturb}/${RUN_ID}"
TRAIN_RUN_DIR="${PIPELINE_DIR}/training"

mkdir -p "${PIPELINE_DIR}" "${LOG_DIR}" "${TRAIN_RUN_DIR}"

common_export="ALL,REPO_ROOT=${REPO_ROOT},PLUGS_ROOT=${PLUGS_ROOT},PIPELINE_DIR=${PIPELINE_DIR},DFODE_DCU_PYTHON=${DFODE_DCU_PYTHON},CASE_FILE=${CASE_FILE},DATA_FILE=${DATA_FILE},SPECIES_FILE=${SPECIES_FILE},SNAPSHOT_CONFIG=${SNAPSHOT_CONFIG},DFODE_MECHANISM=${DFODE_MECHANISM}"

data_job_id="$(
    sbatch --parsable \
        --output="${LOG_DIR}/%x-%j.out" \
        --error="${LOG_DIR}/%x-%j.err" \
        --export="${common_export}" \
        "${DATA_JOB}"
)"

train_job_id="$(
    sbatch --parsable \
        --dependency="afterok:${data_job_id}" \
        --output="${LOG_DIR}/%x-%j.out" \
        --error="${LOG_DIR}/%x-%j.err" \
        --export="${common_export},RUN_ID=${RUN_ID}_train,RUN_DIR=${TRAIN_RUN_DIR},DFODE_OUTPUT_PATH=${TRAIN_RUN_DIR}/checkpoint.pt,DFODE_TRAIN_DATA=${PIPELINE_DIR}/data/train-pairs.h5,DFODE_VALIDATION_DATA=${PIPELINE_DIR}/data/validation-pairs.h5,DFODE_VARIANT=neural-patankar,DFODE_EPOCHS=${DFODE_EPOCHS:-300},DFODE_BATCH_SIZE=${DFODE_BATCH_SIZE:-1024},DFODE_SEED=${DFODE_SEED:-260624}" \
        "${TRAIN_JOB}"
)"

gate_job_id="$(
    sbatch --parsable \
        --dependency="afterok:${train_job_id}" \
        --output="${LOG_DIR}/%x-%j.out" \
        --error="${LOG_DIR}/%x-%j.err" \
        --export="${common_export},TRAIN_RUN_DIR=${TRAIN_RUN_DIR}" \
        "${GATE_JOB}"
)"

cat > "${PIPELINE_DIR}/submitted.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "pipeline_dir": "${PIPELINE_DIR}",
  "data_job_id": "${data_job_id}",
  "training_job_id": "${train_job_id}",
  "gate_job_id": "${gate_job_id}",
  "perturbation_rounds": ${DFODE_PERTURBATION_ROUNDS:-8},
  "data_shards": ${DFODE_DATA_SHARDS:-16},
  "data_parallel_jobs": ${DFODE_DATA_PARALLEL_JOBS:-16}
}
EOF

printf 'Pipeline: %s\n' "${PIPELINE_DIR}"
printf 'Data job: %s\n' "${data_job_id}"
printf 'Training job: %s\n' "${train_job_id}"
printf 'Offline gate job: %s\n' "${gate_job_id}"

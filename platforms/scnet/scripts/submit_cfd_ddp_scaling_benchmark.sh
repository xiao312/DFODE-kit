#!/bin/bash
set -euo pipefail

: "${DFODE_PROJECT_ROOT:?Set DFODE_PROJECT_ROOT}"
: "${DFODE_TRAIN_SOURCE:?Set DFODE_TRAIN_SOURCE}"
: "${DFODE_VALIDATION_SOURCE:?Set DFODE_VALIDATION_SOURCE}"
: "${DFODE_MECHANISM:?Set DFODE_MECHANISM}"

SOURCE_ROOT="${DFODE_SOURCE_ROOT:-${DFODE_PROJECT_ROOT}/.work/src/DFODE-kit}"
PYTHON="${DFODE_PYTHON:-${DFODE_PROJECT_ROOT}/.work/envs/dtk2604-torch251-py311/bin/python}"
RUN_ID="${DFODE_RUN_ID:-$(date +%y%m%d)_ddp_scaling}"
BENCHMARK_ROOT="${DFODE_BENCHMARK_ROOT:-${DFODE_PROJECT_ROOT}/.work/benchmarks/${RUN_ID}}"
GLOBAL_BATCH_SIZE="${DFODE_GLOBAL_BATCH_SIZE:-4096}"
EPOCHS="${DFODE_EPOCHS:-20}"
COUNTS="${DFODE_DCU_COUNTS:-4,8,16,32}"
mkdir -p "${BENCHMARK_ROOT}" "${DFODE_PROJECT_ROOT}/.work/logs"

IFS="," read -r -a dcu_counts <<< "${COUNTS}"
job_ids=()
: > "${BENCHMARK_ROOT}/jobs.tsv"
for dcu_count in "${dcu_counts[@]}"; do
    if (( dcu_count % 4 != 0 )); then
        echo "DCU count must be divisible by four: ${dcu_count}" >&2
        exit 2
    fi
    if (( GLOBAL_BATCH_SIZE % dcu_count != 0 )); then
        echo "Global batch must be divisible by DCU count" >&2
        exit 2
    fi
    node_count=$((dcu_count / 4))
    local_batch=$((GLOBAL_BATCH_SIZE / dcu_count))
    run_dir="${BENCHMARK_ROOT}/dcu-${dcu_count}"
    mkdir -p "${run_dir}"
    job_id="$(
        sbatch --parsable \
            --nodes="${node_count}" \
            --export="ALL,DFODE_PROJECT_ROOT=${DFODE_PROJECT_ROOT},DFODE_SOURCE_ROOT=${SOURCE_ROOT},DFODE_PYTHON=${PYTHON},DFODE_TRAIN_SOURCE=${DFODE_TRAIN_SOURCE},DFODE_VALIDATION_SOURCE=${DFODE_VALIDATION_SOURCE},DFODE_MECHANISM=${DFODE_MECHANISM},DFODE_CHECKPOINT=${run_dir}/checkpoint.pt,DFODE_EPOCHS=${EPOCHS},DFODE_BATCH_SIZE_PER_DEVICE=${local_batch},DFODE_LEARNING_RATE=${DFODE_LEARNING_RATE:-1e-3},DFODE_MINIMUM_LEARNING_RATE=${DFODE_MINIMUM_LEARNING_RATE:-1e-5},DFODE_DCU_COUNT=4,DFODE_SEED=${DFODE_SEED:-260624},DFODE_LOG_EVERY=${DFODE_LOG_EVERY:-10},DFODE_CHECKPOINT_EVERY=${DFODE_CHECKPOINT_EVERY:-10}" \
            "${SOURCE_ROOT}/platforms/scnet/slurm/run_cfd_perturb_distributed_train.slurm"
    )"
    job_ids+=("${job_id}")
    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${dcu_count}" "${node_count}" "${local_batch}" "${job_id}" "${run_dir}" \
        | tee -a "${BENCHMARK_ROOT}/jobs.tsv"
done

dependency="$(
    IFS=:
    echo "${job_ids[*]}"
)"
summary_counts="${COUNTS//,/:}"
summary_job="$(
    sbatch --parsable \
        --dependency="afterok:${dependency}" \
        --export="ALL,DFODE_SOURCE_ROOT=${SOURCE_ROOT},DFODE_PYTHON=${PYTHON},DFODE_BENCHMARK_ROOT=${BENCHMARK_ROOT},DFODE_DCU_COUNTS=${summary_counts}" \
        "${SOURCE_ROOT}/platforms/scnet/slurm/run_cfd_ddp_scaling_summary.slurm"
)"
printf "summary_job=%s\nbenchmark_root=%s\n" \
    "${summary_job}" "${BENCHMARK_ROOT}" \
    | tee -a "${BENCHMARK_ROOT}/jobs.tsv"

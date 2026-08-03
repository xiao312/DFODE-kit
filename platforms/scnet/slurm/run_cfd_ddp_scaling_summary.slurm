#!/bin/bash
#SBATCH --job-name=dfode-scale-report
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}

set -euo pipefail

: "${DFODE_SOURCE_ROOT:?Set DFODE_SOURCE_ROOT}"
: "${DFODE_BENCHMARK_ROOT:?Set DFODE_BENCHMARK_ROOT}"

PYTHON="${DFODE_PYTHON:-python3}"
mkdir -p "${DFODE_BENCHMARK_ROOT}"
"${PYTHON}" "${DFODE_SOURCE_ROOT}/scripts/summarize_ddp_scaling.py" \
    --benchmark-root "${DFODE_BENCHMARK_ROOT}" \
    --dcu-counts "${DFODE_DCU_COUNTS:-4,8,16,32}" \
    --output "${DFODE_BENCHMARK_ROOT}/scaling-report.json"

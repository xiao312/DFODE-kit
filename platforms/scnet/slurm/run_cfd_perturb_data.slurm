#!/usr/bin/env bash
#SBATCH --job-name=dfode-data
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${PLUGS_ROOT:?PLUGS_ROOT is required}"
: "${PIPELINE_DIR:?PIPELINE_DIR is required}"
: "${DFODE_DCU_PYTHON:?DFODE_DCU_PYTHON is required}"
: "${CASE_FILE:?CASE_FILE is required}"
: "${DATA_FILE:?DATA_FILE is required}"
: "${SPECIES_FILE:?SPECIES_FILE is required}"
: "${SNAPSHOT_CONFIG:?SNAPSHOT_CONFIG is required}"
: "${DFODE_MECHANISM:?DFODE_MECHANISM is required}"

SHARDS="${DFODE_DATA_SHARDS:-16}"
PARALLEL_JOBS="${DFODE_DATA_PARALLEL_JOBS:-16}"
PERTURBATION_ROUNDS="${DFODE_PERTURBATION_ROUNDS:-8}"
SEED="${DFODE_SEED:-260624}"
DT="${DFODE_CHEMISTRY_DT:-1e-7}"

mkdir -p "${PIPELINE_DIR}/data"/{snapshots,shards,augmented,pairs,summaries}
cd "${PIPELINE_DIR}/data"
exec > >(tee "${PIPELINE_DIR}/data/run.log") 2>&1

eval "$(/usr/bin/modulecmd bash purge)" || true
eval "$(/usr/bin/modulecmd bash load compiler/dtk/26.04)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY="${DFODE_DCU_PYTHON}"
CLI=(-c 'from dfode_kit.cli.main import main; main()')

"${PY}" "${PLUGS_ROOT}/connectors/ansys-fluent/python/export_cell_snapshot.py" \
    --case "${CASE_FILE}" \
    --data "${DATA_FILE}" \
    --species "${SPECIES_FILE}" \
    --config "${SNAPSHOT_CONFIG}" \
    --output snapshots/full-original.h5

"${PY}" "${CLI[@]}" split-cfd-snapshot \
    --snapshot snapshots/full-original.h5 \
    --train-output snapshots/development-original.h5 \
    --validation-output snapshots/test-original.h5 \
    --validation-fraction 0.15 \
    --seed "${SEED}" \
    --mode random \
    --summary summaries/development-test-split.json

"${PY}" "${CLI[@]}" split-cfd-snapshot \
    --snapshot snapshots/development-original.h5 \
    --train-output snapshots/train-original.h5 \
    --validation-output snapshots/validation-original.h5 \
    --validation-fraction 0.17647058823529413 \
    --seed "$((SEED + 1))" \
    --mode random \
    --summary summaries/train-validation-split.json

for split in train validation test; do
    "${PY}" "${REPO_ROOT}/scripts/shard_cfd_snapshot.py" \
        --source "snapshots/${split}-original.h5" \
        --output-dir "shards/${split}" \
        --prefix "${split}" \
        --shards "${SHARDS}" \
        --summary "summaries/${split}-shards.json"
done

export PY REPO_ROOT DFODE_MECHANISM DT SEED SHARDS PERTURBATION_ROUNDS
run_shard() {
    local shard_number="$1"
    local index
    index="$(printf '%04d' "${shard_number}")"
    local train_snapshot="shards/train/train-${index}.h5"
    local augmented_snapshot="augmented/train-${index}.h5"

    "${PY}" "${REPO_ROOT}/scripts/perturb_cfd_snapshot.py" \
        --source "${train_snapshot}" \
        --reference snapshots/train-original.h5 \
        --mechanism "${DFODE_MECHANISM}" \
        --output "${augmented_snapshot}" \
        --summary "summaries/perturb-${index}.json" \
        --rounds "${PERTURBATION_ROUNDS}" \
        --seed "${SEED}" \
        --shard-index "${shard_number}" \
        --include-original

    "${PY}" -c 'from dfode_kit.cli.main import main; main()' \
        generate-cfd-one-step-pairs \
        --snapshot "${augmented_snapshot}" \
        --mech "${DFODE_MECHANISM}" \
        --output "pairs/train-${index}.h5" \
        --dt "${DT}" \
        --split train \
        --energy on \
        --summary "summaries/train-pairs-${index}.json"

    for split in validation test; do
        "${PY}" -c 'from dfode_kit.cli.main import main; main()' \
            generate-cfd-one-step-pairs \
            --snapshot "shards/${split}/${split}-${index}.h5" \
            --mech "${DFODE_MECHANISM}" \
            --output "pairs/${split}-${index}.h5" \
            --dt "${DT}" \
            --split "${split}" \
            --energy on \
            --summary "summaries/${split}-pairs-${index}.json"
    done
}
export -f run_shard

seq 0 "$((SHARDS - 1))" \
    | xargs -n 1 -P "${PARALLEL_JOBS}" bash -c 'run_shard "$1"' _

for split in train validation test; do
    "${PY}" "${REPO_ROOT}/scripts/merge_interval_pairs.py" \
        --input-dir pairs \
        --pattern "${split}-????.h5" \
        --output "${split}-pairs.h5" \
        --summary "summaries/${split}-merge.json"
done

cat > success.json <<EOF
{
  "status": "passed",
  "seed": ${SEED},
  "perturbation_rounds": ${PERTURBATION_ROUNDS},
  "shards": ${SHARDS},
  "parallel_jobs": ${PARALLEL_JOBS},
  "dt": ${DT}
}
EOF

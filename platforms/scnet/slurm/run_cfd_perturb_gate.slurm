#!/usr/bin/env bash
#SBATCH --job-name=dfode-gate
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${PIPELINE_DIR:?PIPELINE_DIR is required}"
: "${TRAIN_RUN_DIR:?TRAIN_RUN_DIR is required}"
: "${DFODE_DCU_PYTHON:?DFODE_DCU_PYTHON is required}"
: "${DFODE_MECHANISM:?DFODE_MECHANISM is required}"

cd "${PIPELINE_DIR}"
mkdir -p artifact-dense artifact gate
exec > >(tee "${PIPELINE_DIR}/gate/run.log") 2>&1

eval "$(/usr/bin/modulecmd bash purge)" || true
eval "$(/usr/bin/modulecmd bash load compiler/dtk/26.04)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

PY="${DFODE_DCU_PYTHON}"
rm -rf artifact-dense artifact
"${PY}" "${REPO_ROOT}/scripts/export_fluent_artifact.py" \
    --checkpoint "${TRAIN_RUN_DIR}/checkpoint.pt" \
    --mechanism "${DFODE_MECHANISM}" \
    --output artifact-dense
"${PY}" "${REPO_ROOT}/scripts/sparsify_fluent_patankar_artifact.py" \
    --input artifact-dense \
    --output artifact
"${PY}" "${REPO_ROOT}/scripts/add_fluent_artifact_domain.py" \
    --artifact-dir artifact \
    --dataset "${PIPELINE_DIR}/data/train-pairs.h5"
"${PY}" "${REPO_ROOT}/scripts/evaluate_fluent_delta_gate.py" \
    --artifact-dir artifact \
    --source "${PIPELINE_DIR}/data/test-pairs.h5" \
    --output gate/fluent-original-test.json \
    --device cuda:0

cat > gate/success.json <<EOF
{
  "status": "passed",
  "artifact": "${PIPELINE_DIR}/artifact",
  "test_report": "${PIPELINE_DIR}/gate/fluent-original-test.json"
}
EOF

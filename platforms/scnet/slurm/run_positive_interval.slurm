#!/usr/bin/env bash
#SBATCH --job-name=dfode-positive
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${DFODE_DCU_PYTHON:?DFODE_DCU_PYTHON is required}"
: "${DFODE_TRAIN_DATA:?DFODE_TRAIN_DATA is required}"
: "${DFODE_VALIDATION_DATA:?DFODE_VALIDATION_DATA is required}"
: "${DFODE_MECHANISM:?DFODE_MECHANISM is required}"

DFODE_VARIANT="${DFODE_VARIANT:-neural-patankar}"
DFODE_EPOCHS="${DFODE_EPOCHS:-300}"
DFODE_BATCH_SIZE="${DFODE_BATCH_SIZE:-1024}"
DFODE_HIDDEN_DIM="${DFODE_HIDDEN_DIM:-128}"
DFODE_LATENT_DIM="${DFODE_LATENT_DIM:-32}"
DFODE_LR="${DFODE_LR:-1e-3}"
DFODE_TRANSFORM_ALPHA="${DFODE_TRANSFORM_ALPHA:-0.1}"
DFODE_DELTA_LOSS_WEIGHT="${DFODE_DELTA_LOSS_WEIGHT:-0.1}"
DFODE_EXTENT_SCALE="${DFODE_EXTENT_SCALE:-1e-4}"
DFODE_POSITIVITY_FLOOR="${DFODE_POSITIVITY_FLOOR:-0}"
DFODE_SEED="${DFODE_SEED:-260624}"
DFODE_LOG_EVERY="${DFODE_LOG_EVERY:-10}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

export PYTHONHASHSEED="${DFODE_SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

eval "$(/usr/bin/modulecmd bash purge)" || true
eval "$(/usr/bin/modulecmd bash load compiler/dtk/26.04)"

if ! "${DFODE_DCU_PYTHON}" -c 'import cantera, h5py' >/dev/null 2>&1; then
    "${DFODE_DCU_PYTHON}" -m pip install \
        --index-url "${PYPI_INDEX}" \
        "cantera==3.2.0" \
        "h5py>=3.11"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cat > training-config.json <<EOF
{
  "train_source": "${DFODE_TRAIN_DATA}",
  "validation_source": "${DFODE_VALIDATION_DATA}",
  "mechanism": "${DFODE_MECHANISM}",
  "variant": "${DFODE_VARIANT}",
  "epochs": ${DFODE_EPOCHS},
  "batch_size": ${DFODE_BATCH_SIZE},
  "hidden_dim": ${DFODE_HIDDEN_DIM},
  "latent_dim": ${DFODE_LATENT_DIM},
  "learning_rate": ${DFODE_LR},
  "transform_alpha": ${DFODE_TRANSFORM_ALPHA},
  "delta_loss_weight": ${DFODE_DELTA_LOSS_WEIGHT},
  "extent_scale": ${DFODE_EXTENT_SCALE},
  "positivity_floor": ${DFODE_POSITIVITY_FLOOR},
  "seed": ${DFODE_SEED},
  "deterministic": true,
  "device": "cuda:0"
}
EOF

"${DFODE_DCU_PYTHON}" - <<'PY'
import cantera
import h5py
import torch

print("torch", torch.__version__)
print("cantera", cantera.__version__)
print("h5py", h5py.__version__)
print("dcu_available", torch.cuda.is_available())
print("dcu_name", torch.cuda.get_device_name(0))
PY

start_epoch="$(date +%s)"
"${DFODE_DCU_PYTHON}" -c \
    'from dfode_kit.cli.main import main; main()' \
    train-positive-interval \
    --train-source "${DFODE_TRAIN_DATA}" \
    --validation-source "${DFODE_VALIDATION_DATA}" \
    --mech "${DFODE_MECHANISM}" \
    --output "${RUN_DIR}/checkpoint.pt" \
    --variant "${DFODE_VARIANT}" \
    --device cuda:0 \
    --epochs "${DFODE_EPOCHS}" \
    --batch-size "${DFODE_BATCH_SIZE}" \
    --learning-rate "${DFODE_LR}" \
    --hidden-dim "${DFODE_HIDDEN_DIM}" \
    --latent-dim "${DFODE_LATENT_DIM}" \
    --transform-alpha "${DFODE_TRANSFORM_ALPHA}" \
    --delta-loss-weight "${DFODE_DELTA_LOSS_WEIGHT}" \
    --extent-scale "${DFODE_EXTENT_SCALE}" \
    --positivity-floor "${DFODE_POSITIVITY_FLOOR}" \
    --seed "${DFODE_SEED}" \
    --log-every "${DFODE_LOG_EVERY}"
end_epoch="$(date +%s)"

"${DFODE_DCU_PYTHON}" -c \
    'from dfode_kit.cli.main import main; main()' \
    evaluate-positive-interval \
    --checkpoint "${RUN_DIR}/checkpoint.pt" \
    --source "${DFODE_VALIDATION_DATA}" \
    --mech "${DFODE_MECHANISM}" \
    --output "${RUN_DIR}/evaluation.json" \
    --method none \
    --device cuda:0

cat > success.json <<EOF
{
  "run_id": "${RUN_ID}",
  "checkpoint": "${RUN_DIR}/checkpoint.pt",
  "evaluation": "${RUN_DIR}/evaluation.json",
  "elapsed_seconds": $((end_epoch - start_epoch)),
  "status": "passed"
}
EOF

printf 'Training completed in %s seconds.\n' "$((end_epoch - start_epoch))"

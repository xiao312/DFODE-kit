#!/usr/bin/env bash
#SBATCH --job-name=dfode_train_dcu
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
: "${DFODE_WHEEL_DIR:?DFODE_WHEEL_DIR is required}"
: "${DFODE_TRAIN_DATA:?DFODE_TRAIN_DATA is required}"
: "${DFODE_MECHANISM:?DFODE_MECHANISM is required}"
: "${DFODE_OUTPUT_PATH:?DFODE_OUTPUT_PATH is required}"

DFODE_EPOCHS="${DFODE_EPOCHS:-1000}"
DFODE_BATCH_SIZE="${DFODE_BATCH_SIZE:-4096}"
DFODE_LATENT_DIM="${DFODE_LATENT_DIM:-32}"
DFODE_HIDDEN_DIM="${DFODE_HIDDEN_DIM:-256}"
DFODE_LR="${DFODE_LR:-1e-3}"
DFODE_TRANSFORM_ALPHA="${DFODE_TRANSFORM_ALPHA:-0.1}"
DFODE_SPECIES_LOSS_WEIGHT="${DFODE_SPECIES_LOSS_WEIGHT:-0.1}"
DFODE_LOG_EVERY_EPOCHS="${DFODE_LOG_EVERY_EPOCHS:-25}"
DFODE_SEED="${DFODE_SEED:-20260728}"

mkdir -p "${RUN_DIR}" "$(dirname "${DFODE_OUTPUT_PATH}")"
cd "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

export PYTHONHASHSEED="${DFODE_SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

printf 'run_id=%s\n' "${RUN_ID}"
printf 'host=%s\n' "$(hostname)"
printf 'glibc=%s\n' "$(ldd --version | head -n 1)"

eval "$(/usr/bin/modulecmd bash purge)" || true
eval "$(/usr/bin/modulecmd bash load compiler/dtk/26.04)"

"${DFODE_DCU_PYTHON}" -m pip install \
    --quiet \
    --no-index \
    --find-links "${DFODE_WHEEL_DIR}" \
    "cantera==3.2.0" \
    "h5py==3.11.0"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cat > training-config.json <<EOF
{
  "source": "${DFODE_TRAIN_DATA}",
  "mechanism": "${DFODE_MECHANISM}",
  "output": "${DFODE_OUTPUT_PATH}",
  "seed": ${DFODE_SEED},
  "deterministic": true,
  "shuffle_algorithm": "numpy-pcg64-seedsequence-v1",
  "epochs": ${DFODE_EPOCHS},
  "batch_size": ${DFODE_BATCH_SIZE},
  "latent_dim": ${DFODE_LATENT_DIM},
  "hidden_dim": ${DFODE_HIDDEN_DIM},
  "learning_rate": ${DFODE_LR},
  "transform_alpha": ${DFODE_TRANSFORM_ALPHA},
  "species_loss_weight": ${DFODE_SPECIES_LOSS_WEIGHT},
  "loss": "mae",
  "flux_mode": "signed-power",
  "model_variant": "stoich",
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
    train-stoich-interval \
    --source "${DFODE_TRAIN_DATA}" \
    --output "${DFODE_OUTPUT_PATH}" \
    --mech "${DFODE_MECHANISM}" \
    --seed "${DFODE_SEED}" \
    --deterministic \
    --latent-dim "${DFODE_LATENT_DIM}" \
    --hidden-dim "${DFODE_HIDDEN_DIM}" \
    --epochs "${DFODE_EPOCHS}" \
    --batch-size "${DFODE_BATCH_SIZE}" \
    --lr "${DFODE_LR}" \
    --transform-alpha "${DFODE_TRANSFORM_ALPHA}" \
    --species-loss-weight "${DFODE_SPECIES_LOSS_WEIGHT}" \
    --loss-kind mae \
    --flux-mode signed-power \
    --model-variant stoich \
    --log-every-epochs "${DFODE_LOG_EVERY_EPOCHS}" \
    --device cuda:0
end_epoch="$(date +%s)"

cat > success.json <<EOF
{
  "run_id": "${RUN_ID}",
  "checkpoint": "${DFODE_OUTPUT_PATH}",
  "seed": ${DFODE_SEED},
  "deterministic": true,
  "elapsed_seconds": $((end_epoch - start_epoch)),
  "status": "passed"
}
EOF

printf 'Training completed in %s seconds.\n' "$((end_epoch - start_epoch))"

#!/usr/bin/env bash
set -euo pipefail

: "${SNAPSHOT:?SNAPSHOT is required}"
: "${MECH:?MECH is required}"
: "${RUN_DIR:?RUN_DIR is required}"

SEED="${SEED:-20260728}"
STEPS="${STEPS:-96}"
MIN_TIME="${MIN_TIME:-1e-9}"
MAX_TIME="${MAX_TIME:-1e-3}"
MAX_PAIRS_PER_TRAJECTORY_PER_BIN="${MAX_PAIRS_PER_TRAJECTORY_PER_BIN:-2}"
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda:0}"
BASE_TRAIN_PAIRS="${BASE_TRAIN_PAIRS:-}"
BASE_VAL_PAIRS="${BASE_VAL_PAIRS:-}"

mkdir -p "${RUN_DIR}"

dfode-kit generate-cfd-conditioned-sequences \
    --snapshot "${SNAPSHOT}" \
    --mech "${MECH}" \
    --train-output "${RUN_DIR}/cfd_train_sequences.h5" \
    --validation-output "${RUN_DIR}/cfd_val_sequences.h5" \
    --summary "${RUN_DIR}/cfd_sequence_summary.json" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --min-time "${MIN_TIME}" \
    --max-time "${MAX_TIME}"

dfode-kit generate-interval-pairs \
    --source "${RUN_DIR}/cfd_train_sequences.h5" \
    --output "${RUN_DIR}/cfd_train_pairs.h5" \
    --min-dt "${MIN_TIME}" \
    --max-dt "${MAX_TIME}" \
    --max-pairs-per-trajectory-per-bin \
        "${MAX_PAIRS_PER_TRAJECTORY_PER_BIN}" \
    --seed "${SEED}" \
    --split train

dfode-kit generate-interval-pairs \
    --source "${RUN_DIR}/cfd_val_sequences.h5" \
    --output "${RUN_DIR}/cfd_val_pairs.h5" \
    --min-dt "${MIN_TIME}" \
    --max-dt "${MAX_TIME}" \
    --max-pairs-per-trajectory-per-bin \
        "${MAX_PAIRS_PER_TRAJECTORY_PER_BIN}" \
    --seed "${SEED}" \
    --split val

TRAIN_SOURCE="${RUN_DIR}/cfd_train_pairs.h5"
VAL_SOURCE="${RUN_DIR}/cfd_val_pairs.h5"
if [[ -n "${BASE_TRAIN_PAIRS}" ]]; then
    dfode-kit merge-interval-pairs \
        --source "${BASE_TRAIN_PAIRS}" --label reactor-suite \
        --source "${TRAIN_SOURCE}" --label fluent-cfd \
        --output "${RUN_DIR}/train_pairs.h5" \
        --split train
    TRAIN_SOURCE="${RUN_DIR}/train_pairs.h5"
fi
if [[ -n "${BASE_VAL_PAIRS}" ]]; then
    dfode-kit merge-interval-pairs \
        --source "${BASE_VAL_PAIRS}" --label reactor-suite \
        --source "${VAL_SOURCE}" --label fluent-cfd \
        --output "${RUN_DIR}/val_pairs.h5" \
        --split val
    VAL_SOURCE="${RUN_DIR}/val_pairs.h5"
fi

printf '%s\n' "${VAL_SOURCE}" > "${RUN_DIR}/validation_dataset.txt"
dfode-kit train-stoich-interval \
    --source "${TRAIN_SOURCE}" \
    --output "${RUN_DIR}/gri30_cfd_conditioned.pt" \
    --mech "${MECH}" \
    --seed "${SEED}" \
    --deterministic \
    --latent-dim 32 \
    --hidden-dim 256 \
    --epochs "${EPOCHS}" \
    --batch-size 4096 \
    --transform-alpha 0.1 \
    --species-loss-weight 0.1 \
    --loss-kind mae \
    --flux-mode signed-power \
    --device "${DEVICE}" \
    2>&1 | tee "${RUN_DIR}/train.log"

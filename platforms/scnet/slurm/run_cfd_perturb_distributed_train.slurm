#!/bin/bash
#SBATCH --job-name=dfode-ddp
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH --output=.work/logs/%x-%j.out
#SBATCH --error=.work/logs/%x-%j.err

set -euo pipefail

: "${DFODE_PROJECT_ROOT:?Set DFODE_PROJECT_ROOT}"
: "${DFODE_TRAIN_SOURCE:?Set DFODE_TRAIN_SOURCE}"
: "${DFODE_VALIDATION_SOURCE:?Set DFODE_VALIDATION_SOURCE}"
: "${DFODE_MECHANISM:?Set DFODE_MECHANISM}"
: "${DFODE_CHECKPOINT:?Set DFODE_CHECKPOINT}"

PYTHON="${DFODE_PYTHON:-${DFODE_PROJECT_ROOT}/.work/envs/dtk2604-torch251-py311/bin/python}"
SOURCE_ROOT="${DFODE_SOURCE_ROOT:-${DFODE_PROJECT_ROOT}/.work/src/DFODE-kit}"
DCU_COUNT="${DFODE_DCU_COUNT:-4}"
RUN_DIR="$(dirname "${DFODE_CHECKPOINT}")"
mkdir -p "${RUN_DIR}" "${DFODE_PROJECT_ROOT}/.work/logs"

module purge
module load compiler/dtk/26.04
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${DFODE_OMP_NUM_THREADS:-2}"
export HIP_VISIBLE_DEVICES="$(seq -s, 0 $((DCU_COUNT - 1)))"
export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export DFODE_PYTHON="${PYTHON}"
export DFODE_SOURCE_ROOT="${SOURCE_ROOT}"
export MASTER_ADDR
MASTER_ADDR="${DFODE_MASTER_ADDR:-$(
    /sbin/ifconfig ib0 \
        | awk '$1 == "inet" {print $2; exit}'
)}"
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to determine the batch node ib0 address" >&2
    exit 2
fi
export MASTER_PORT="${DFODE_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export NCCL_SOCKET_IFNAME="${DFODE_NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${DFODE_GLOO_SOCKET_IFNAME:-ib0}"
export NCCL_IB_DISABLE="${DFODE_NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${DFODE_NCCL_DEBUG:-WARN}"

cd "${SOURCE_ROOT}"
export DFODE_TRAIN_SOURCE DFODE_VALIDATION_SOURCE DFODE_MECHANISM
export DFODE_CHECKPOINT RUN_DIR DCU_COUNT
export DFODE_EPOCHS="${DFODE_EPOCHS:-2000}"
export DFODE_BATCH_SIZE_PER_DEVICE="${DFODE_BATCH_SIZE_PER_DEVICE:-1024}"
export DFODE_LEARNING_RATE="${DFODE_LEARNING_RATE:-1e-3}"
export DFODE_MINIMUM_LEARNING_RATE="${DFODE_MINIMUM_LEARNING_RATE:-1e-5}"
export DFODE_HIDDEN_DIM="${DFODE_HIDDEN_DIM:-128}"
export DFODE_LATENT_DIM="${DFODE_LATENT_DIM:-32}"
export DFODE_TRANSFORM_ALPHA="${DFODE_TRANSFORM_ALPHA:-0.1}"
export DFODE_DELTA_LOSS_WEIGHT="${DFODE_DELTA_LOSS_WEIGHT:-0.1}"
export DFODE_EXTENT_SCALE="${DFODE_EXTENT_SCALE:-1e-4}"
export DFODE_SEED="${DFODE_SEED:-260624}"
export DFODE_LOG_EVERY="${DFODE_LOG_EVERY:-10}"
export DFODE_CHECKPOINT_EVERY="${DFODE_CHECKPOINT_EVERY:-10}"

srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    bash -c '
        "${DFODE_PYTHON}" -m torch.distributed.run \
            --nnodes="${SLURM_NNODES}" \
            --nproc_per_node="${DCU_COUNT}" \
            --node_rank="${SLURM_NODEID}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            "${DFODE_SOURCE_ROOT}/scripts/train_positive_interval_distributed.py" \
            --train-source "${DFODE_TRAIN_SOURCE}" \
            --validation-source "${DFODE_VALIDATION_SOURCE}" \
            --mechanism "${DFODE_MECHANISM}" \
            --output "${DFODE_CHECKPOINT}" \
            --final-output "${RUN_DIR}/final.pt" \
            --resume-output "${RUN_DIR}/last.pt" \
            --epochs "${DFODE_EPOCHS}" \
            --batch-size-per-device "${DFODE_BATCH_SIZE_PER_DEVICE}" \
            --learning-rate "${DFODE_LEARNING_RATE}" \
            --minimum-learning-rate "${DFODE_MINIMUM_LEARNING_RATE}" \
            --hidden-dim "${DFODE_HIDDEN_DIM}" \
            --latent-dim "${DFODE_LATENT_DIM}" \
            --transform-alpha "${DFODE_TRANSFORM_ALPHA}" \
            --delta-loss-weight "${DFODE_DELTA_LOSS_WEIGHT}" \
            --extent-scale "${DFODE_EXTENT_SCALE}" \
            --seed "${DFODE_SEED}" \
            --log-every "${DFODE_LOG_EVERY}" \
            --checkpoint-every "${DFODE_CHECKPOINT_EVERY}"
    ' 2>&1 | tee "${RUN_DIR}/train.log"

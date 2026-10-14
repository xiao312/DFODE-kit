#!/bin/sh
# One visible GPU, sequential fits; queue defaults to read-only discovery.
set -eu
mode="${1:---dry-run}"
case "$mode" in --dry-run|--execute) ;; *) exit 2;; esac
: "${CUDA_VISIBLE_DEVICES:?Select one free GPU explicitly}"
: "${PAIRED_PYTHON:?Set the project GPU Python}"
: "${PAIRED_OUTPUT:?Set a new campaign directory}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
failed=0
for seed in 20261011 20261012; do
  sh benchmarks/offline_accuracy/paired/queue.sh "$seed" "$mode" || failed=1
done
exit "$failed"

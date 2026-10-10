#!/bin/sh
# Run from the project root with a project-local GPU Python and new output root.
set -u
seed="$1"
mode="${2:---dry-run}"
python="${PAIRED_PYTHON:?Set the project GPU Python}"
output="${PAIRED_OUTPUT:?Set a new campaign directory}"
case "$seed" in 20261011|20261012) ;; *) exit 2;; esac
case "$mode" in --dry-run) preview="--dry-run";; --execute) preview="";; *) exit 2;; esac
failed=0
for target in state-boxcox gbct; do
  for objective in coordinate increment state; do
    variant="$target-$objective"
    "$python" -m benchmarks.offline_accuracy.paired.run \
      runs/offline-accuracy/20261009/preparation/dataset \
      --audit runs/offline-accuracy/20261009/preparation/audit \
      --base "runs/offline-accuracy/20261009/seed-$seed/training" \
      --seed "$seed" --variant "$variant" \
      --output "$output/seed-$seed/$variant" $preview || failed=1
  done
done
exit "$failed"

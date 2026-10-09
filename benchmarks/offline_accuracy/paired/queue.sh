#!/bin/sh
# Run inside the pinned container: /work read-only, /evidence new campaign only.
set -u
seed="$1"
mode="${2:---dry-run}"
case "$seed" in 20261011|20261012) ;; *) exit 2;; esac
case "$mode" in --dry-run) preview="--dry-run";; --execute) preview="";; *) exit 2;; esac
failed=0
for target in state-boxcox gbct; do
  for objective in coordinate increment state; do
    variant="$target-$objective"
    .venv/bin/python -m benchmarks.offline_accuracy.paired.run \
      runs/offline-accuracy/20261009/preparation/dataset \
      --audit runs/offline-accuracy/20261009/preparation/audit \
      --base "runs/offline-accuracy/20261009/seed-$seed/training" \
      --seed "$seed" --variant "$variant" \
      --output "/evidence/seed-$seed/$variant" $preview || failed=1
  done
done
exit "$failed"

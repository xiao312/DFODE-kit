# SCNET DCU training

SCNET training belongs to `DFODE-kit`. Fluent integration and native inference
remain in `dfode-plugs`.

The launcher requires explicit external paths so datasets, environments, and
checkpoints remain outside the source tree:

```bash
export DFODE_DCU_PYTHON=/path/to/dcu/python
export DFODE_WHEEL_DIR=/path/to/offline/wheels
export DFODE_TRAIN_DATA=/path/to/train_interval_pairs.h5
export DFODE_MECHANISM=/path/to/mechanism.yaml
export DFODE_RUN_ROOT=/path/to/runs
export DFODE_LOG_ROOT=/path/to/logs
export DFODE_SEED=20260728

bash platforms/scnet/scripts/submit_training.sh
```

For the positivity-preserving Neural Patankar baseline, provide both split
datasets and use:

```bash
export DFODE_DCU_PYTHON=/path/to/dcu/python
export DFODE_TRAIN_DATA=/path/to/train_pairs.h5
export DFODE_VALIDATION_DATA=/path/to/validation_pairs.h5
export DFODE_MECHANISM=/path/to/gri30.yaml
export DFODE_RUN_ROOT=/path/to/training-runs
export DFODE_LOG_ROOT=/path/to/logs
export DFODE_SEED=260624

bash platforms/scnet/scripts/submit_positive_interval.sh
```

This job trains the checkpoint and evaluates the held-out data in the same DCU
allocation. Its defaults reproduce the current baseline: Neural Patankar,
signed-power alpha `0.1`, MAE-based combined state/change training,
`delta_loss_weight=0.1`, and `extent_scale=1e-4`.

## CFD random-perturbation pipeline

`submit_cfd_perturb_pipeline.sh` runs the complete CFD-conditioned workflow
through Slurm:

1. Export every Fluent cell from case/data HDF5.
2. Randomly split original cells into 70% training, 15% validation, and 15%
   untouched offline test sets.
3. Apply configurable repeated Xiao-style random perturbations only to the
   training cells.
4. Filter temperature, pressure, nitrogen, composition, and significant
   negative heat-release violations.
5. Generate Cantera labels in parallel shards and merge interval-pair files.
6. Train the Neural Patankar model on a DCU.
7. Export the Fluent artifact and evaluate reaction-increment magnitude,
   direction, sign, positivity, and conservation on untouched Fluent cells.

The default is eight perturbation rounds and 16 parallel data shards. Increase
`DFODE_PERTURBATION_ROUNDS` to grow the data volume without changing the test
set. Generated data remains under the configured `.work/pipelines` root.

For multi-million-state training, use
`scripts/submit_cfd_perturb_scale_pipeline.sh`. Its defaults use 192
independent perturbation rounds, 32 CPU labeling shards, and deterministic
four-DCU distributed training for 2000 epochs. The distributed trainer keeps
physical species states and hard chemistry operations in float64 while neural
inputs and trainable layers remain float32. It uses a reproducible global
shuffle, AdamW with cosine learning-rate annealing, `checkpoint.pt` for the
best validation state, `last.pt` for restart, and `final.pt` for explicit
last-epoch analysis.

Validation and test states are split from the original Fluent field before
augmentation. No target-magnitude or reaction-activity balancing is applied.

## Multi-node DDP scaling

`run_cfd_perturb_distributed_train.slurm` supports one or more four-DCU
SCNET nodes through a Slurm-launched `torchrun` rendezvous. Use
`scripts/submit_cfd_ddp_scaling_benchmark.sh` to compare 4, 8, 16, and 32
DCUs while holding the global batch, seed, data, optimizer, schedule, and
epoch count fixed. The default global batch is 4096, giving per-DCU batches
of 1024, 512, 256, and 128 respectively.

The dependent summary job writes `scaling-report.json` and
`scaling-report.csv` with steady epoch time, aggregate samples per second,
speedup, parallel efficiency, and validation losses. Select the subsequent
full-run DCU count from both throughput and validation behavior rather than
raw wall time alone.

The GRI30 3.61-million-state benchmark measured 4/8/16/32-DCU speedups of
`1.00/1.24/1.28/1.22` at fixed global batch 4096. Eight DCUs are therefore
the default for subsequent full runs because 16 DCUs provide only about four
percent additional throughput at twice the card cost. Set
`DFODE_TRAIN_NODES=4` and `DFODE_BATCH_SIZE_PER_DEVICE=256` to select the
16-DCU time-priority configuration. Do not use 32 DCUs for this model size.

The Slurm job requests one DCU on a glibc 2.31 node, loads DTK 26.04, installs
Cantera and h5py from the offline wheel cache, and launches deterministic
stoichiometric-interval training.

Deterministic mode provides:

- seeded Python, NumPy, Torch, and accelerator RNGs;
- a NumPy PCG64 epoch permutation independent of the accelerator backend;
- deterministic Torch algorithm enforcement;
- disabled TF32 and cuDNN benchmarking;
- a checkpoint hash of the initial parameters.

The same software stack, device type, and driver should reproduce results
bitwise. NVIDIA CUDA and SCNET DTK/HIP are different numerical stacks, so
cross-platform runs are expected to begin with identical parameters and sample
order but are not promised to remain bitwise identical after optimization.

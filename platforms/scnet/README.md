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

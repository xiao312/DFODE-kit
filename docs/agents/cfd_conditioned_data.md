# CFD-conditioned chemistry data

The CFD-conditioned pipeline complements generic 0D reactor sampling with
thermochemical states actually occupied by a CFD calculation.

## Repository boundary

`dfode-plugs` exports solver-specific Fluent fields into the portable
`cfd-cell-snapshot-v1` contract. `DFODE-kit` owns species remapping, physical
validation, Cantera integration, train/validation splitting, interval-pair
construction, dataset merging, and retraining.

## Leakage prevention

The train/validation split is assigned to source CFD cells, stratified by the
snapshot's sampling strata, before trajectories and interval pairs are
generated. All intervals derived from one CFD cell therefore remain in one
split.

## Sampling dimensions

The Fluent adapter stratifies over:

- temperature;
- log-magnitude of the available Fluent reaction-rate proxy;
- composition entropy;
- number of exact-zero species.

This deliberately retains cold inlet states, mixed nonreactive states, hot
pilot/products, active reaction zones, and sparse-composition boundaries.

## Data composition

CFD-conditioned pairs should normally be merged with the existing reactor
suite rather than replacing it. The merged HDF5 dataset records a
`source_dataset_index` for data-composition diagnostics and later weighted
sampling.

## Commands

```bash
dfode-kit generate-cfd-conditioned-sequences ...
dfode-kit generate-interval-pairs ...
dfode-kit merge-interval-pairs ...
dfode-kit train-stoich-interval ...
```

For a reproducible baseline, set `SNAPSHOT`, `MECH`, and `RUN_DIR`, then run
`scripts/run_cfd_conditioned_retrain.sh`.

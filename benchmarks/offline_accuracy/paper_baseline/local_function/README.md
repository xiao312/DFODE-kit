# Fixed-centre trained local corrections

## Responsibility

Test learned local functions with global versus frozen species rank distance.
This is not nearest-label copying, a production change or a new evaluator.
Reuse the continued base, checked 50k library, 20k unseen population and 1023 D
states. Metric calibration retains its 48,976-row exposure. No U/D data fit any
part of the model. Read the registered experiment before each run.

## Model and interfaces

Select 128 common centres: 32 per training snapshot, from source-ID-sorted rows
with the run seed. The centre feature values and identities are identical in both
arms. For species i, compute squared distance q_ik with either all-one weights
or its frozen positive diagonal weights. Set h_i^2 to the median nearest-centre
squared distance on the first 4096 non-centre training rows. A zero median uses
the median strictly positive distance, or 1 if none; record any fallback.
Use phi_ik = exp(-q_ik/(2 h_i^2)) / (1 + sum_k exp(-q_ik/(2 h_i^2))).
The extra denominator 1 is a zero-correction anchor; far from every centre,
the correction tends to zero. Infer c_i = sum_k phi_ik * theta_ik. Learn only
7424 coefficients, initially zero. Decode with the checked fixed-asinh decoder.

Distance and kernels use FP64 arithmetic, with nonnegative clipping of squared
distance solely to remove norm-expansion roundoff. Basis values and learned
coefficients are FP32; labels and loss are FP64. No prediction clipping occurs.
The first comparison fixes Adam, Smooth L1 beta=log(1.1), 6000 updates, batch
2048, cosine learning rate 0.01 to 0.0001, and coefficient L2 penalty 1e-6.
Report training-only curves and use the final checkpoint. No U/D selection.

`run --project <root> --output <new-dir>` is read-only by default; `--execute`
admits work. `worker train` mounts training features/labels only. `worker infer`
mounts states/IDs and frozen base, preprocessing, geometry and coefficients only.
Outputs include model hashes, replay predictions, scores, timing and job records.

## Dependencies, security and verification

`run -> research_contract identity/admission/sandbox + hidden_features.run_worker`;
`worker -> local_function.model + correction_scaling base + adaptive.features`;
`review -> saved physical arrays + pinned physical scorer`. Production modules
do not depend on this module. No secrets or query references enter inference.
Use one idle GPU and stop only our worker on contention. CPU verification is
single-threaded. Large artifacts remain in ignored project runs.

Run `python -m pytest tests/test_local_function.py -q`. Expect deterministic centre
identity, weighted distance parity, zero-correction identity, smooth far-field
decay, finite gradients and saved-model replay tests. Candidate qualification
requires full T/U/D scoring, independent physical recounts and matched end-to-end
base/candidate latency. No automatic production promotion exists.

`postcheck --project <root> --run <completed-run>` checks an existing result,
without fitting. It admits one isolated GPU worker per seed to replay the frozen
base from state-only inputs for all three populations. The trusted controller
compares the saved base arrays outside the sandbox. `review` then independently
recounts tails, gate failures, snapshot scores and damage by original error
margin. These post-fit checks do not change predictions or select checkpoints.

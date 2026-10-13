"""No-reference inference and measured complete cost for bounded base consensus."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json, digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..correction_scaling.model import load_base, base_prediction
from ..adaptive.model import base_preprocessing
from ..hidden_features.prepare import save
from .metrics import bounded_mean, strict_shrink


def load_models(seeds):
    return {seed: load_base(Path("/model") / str(seed), seed) for seed in seeds}


def base_only(models, states, seed):
    model, prep = models[seed]
    return base_prediction(model, base_preprocessing(prep), states, seed)[0]


def predict(models, states, seed, fraction, mode="bounded"):
    other = next(value for value in models if value != seed)
    b, q = base_only(models, states, seed), base_only(models, states, other)
    result=bounded_mean(b, q, fraction)
    if mode=="strict-shrink": result=strict_shrink(b,result,fraction)
    elif mode!="bounded": raise ValueError("Unknown prediction rule")
    return result, b, q


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--seed", type=int, required=True); args = parser.parse_args()
    environment = configure(); started = time.monotonic()
    plan = read_json("/input/config.json"); rows = arrays("/input/states.npz")
    if set(rows) != {"states", "source_indices"}: raise ValueError("Unexpected inference fields")
    models = load_models(plan["seeds"])
    mode=plan.get("mode","bounded")
    prediction, base, other = predict(models, rows["states"], args.seed, plan["fraction"],mode)
    del models
    models = load_models(plan["seeds"])
    replay = predict(models, rows["states"], args.seed, plan["fraction"],mode)
    for actual, expected in zip(replay, (prediction, base, other)): np.testing.assert_array_equal(actual, expected)
    timings = dict(anchor=[], candidate=[])
    states = rows["states"][:1024]
    for repeat in range(6):
        for name in ("anchor", "candidate") if repeat % 2 == 0 else ("candidate", "anchor"):
            torch.cuda.synchronize(); start = time.monotonic()
            if name == "anchor": base_only(models, states, args.seed)
            else: predict(models, states, args.seed, plan["fraction"],mode)
            torch.cuda.synchronize()
            if repeat: timings[name].append(time.monotonic()-start)
    elapsed = time.monotonic()-started
    if elapsed > plan["resources"]["maximum_worker_seconds"]: raise TimeoutError("Finite inference budget exceeded")
    np.savez("/output/predictions.npz", prediction=prediction, base=base, other=other, source_indices=rows["source_indices"])
    save("/output/result.json", dict(status="complete", environment=environment, elapsed_seconds=elapsed,
         exact_reload_replay=True, timings=timings, ratio=float(np.median(timings["candidate"])/np.median(timings["anchor"])),
         prediction_sha256=digest("/output/predictions.npz"), confirmation_opened=False))


if __name__ == "__main__": main()

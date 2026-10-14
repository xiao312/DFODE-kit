"""Restricted worker: training labels OR state-only inference, never both mounts."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json, digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..correction_scaling.train import fit
from ..correction_scaling.model import load_base, cached_prediction, base_prediction
from ..resolved_fit.fit import head
from .model import ARMS, predict, config
from .prepare import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "infer"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    runtime = configure()
    destination = Path("/output")
    cfg = config(args.seed, args.arm)
    started = time.perf_counter()
    if args.mode == "train":
        supplied = read_json("/input/config.json")
        if supplied != cfg:
            raise ValueError("Worker configuration differs")
        rows = arrays("/input/training.npz")
        if args.arm == "P":
            from .preservation import objective
            model, fitted = fit(rows, None, rows["active"], cfg, destination, objective_function=objective)
        else:
            model, fitted = fit(rows, None, rows["active"], cfg, destination)
        if fitted["parameter_count"] != ARMS[args.arm][2]:
            raise ValueError("Parameter control mismatch")
        predicted, coordinate = cached_prediction(model, rows, rows["active"], cfg)
        np.savez(destination / "training-replay.npz", prediction=predicted, coordinate=coordinate)
        save(destination / "result.json", dict(status="complete", config=cfg, environment=runtime,
                                               **fitted, total_seconds=time.perf_counter()-started))
        return
    rows = arrays("/input/states.npz")
    if set(rows) != {"states", "source_indices"}:
        raise ValueError("Inference receives states and identities only")
    base, prep = load_base(Path("/model"), args.seed)
    scales = arrays("/model/hidden-scales.npz")
    model = head(cfg, ARMS[args.arm][0], int(prep["active"].sum()))
    model.load_state_dict(torch.load("/model/weights.pt", weights_only=True, map_location="cuda"))
    model.eval()
    p, c = predict(base, model, prep, scales, rows["states"], cfg)
    again, ac = predict(base, model, prep, scales, rows["states"], cfg)
    np.testing.assert_array_equal(p, again)
    np.testing.assert_array_equal(c, ac)
    np.savez(destination / "predictions.npz", prediction=p, coordinate=c, source_indices=rows["source_indices"])
    # Same state batch, one warmup each, alternating order; includes CPU preparation,
    # frozen forward, head, host transfer and physical decoding.
    sample = rows["states"][:1023]
    times = {"candidate":[], "anchor":[]}
    calls = {"candidate":lambda:predict(base, model, prep, scales, sample, cfg),
             "anchor":lambda:base_prediction(base, {k[5:]:v for k,v in prep.items() if k.startswith("base_")}, sample, args.seed)}
    for call in calls.values():
        call()
    for repeat in range(10):
        for name in (("candidate", "anchor") if repeat % 2 == 0 else ("anchor", "candidate")):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            calls[name]()
            torch.cuda.synchronize()
            times[name].append(time.perf_counter()-tick)
    save(destination / "result.json", dict(status="complete", exact_repeat=True, environment=runtime,
         states=len(p), timing_states=len(sample), timings=times, config=cfg,
         prediction_sha256=digest(destination / "predictions.npz"), total_seconds=time.perf_counter()-started))


if __name__ == "__main__":
    main()

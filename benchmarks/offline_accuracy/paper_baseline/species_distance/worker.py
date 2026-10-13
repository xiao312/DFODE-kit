"""Isolated GPU search; frozen weights are the only training-derived artifact."""
import time
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..local_predictability.neighbors import search
from .metric import scale_features


def main():
    environment = configure()
    rows = arrays("/input/features.npz")
    if set(rows) != {"library_features","query_features","library_ids","query_ids","weights"}:
        raise ValueError("Worker accepts features, IDs and frozen weights only")
    plan = read_json("/input/plan.json")
    if rows["weights"].shape != (58,119):
        raise ValueError("Unexpected metric shape")
    output, timings = {}, []
    weights = [np.ones(119)] + list(rows["weights"])
    for index, weight in enumerate(weights):
        start = time.perf_counter()
        space = "global" if index == 0 else f"species_{index-1}"
        result = search(scale_features(rows["library_features"],weight),
                        scale_features(rows["query_features"],weight),
                        rows["library_ids"],rows["query_ids"],plan["libraries"])
        torch.cuda.synchronize()
        for size, values in result.items():
            for key,value in values.items():
                output[f"{space}_{size}_{key}"] = value
        timings.append(dict(space=space,seconds=time.perf_counter()-start))
        save("/output/progress.json",dict(completed_searches=index+1,total_searches=len(weights),timings=timings))
        print(f"Completed {space}: {index+1}/{len(weights)}",flush=True)
    np.savez("/output/neighbors.npz",**output)
    save("/output/result.json",dict(status="complete",environment=environment,timings=timings,
         query_labels_received=False,training_artifact="frozen species weights"))


if __name__ == "__main__":
    main()

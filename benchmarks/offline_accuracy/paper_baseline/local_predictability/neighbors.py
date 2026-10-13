"""Feature-only GPU neighbor search. No target arrays enter this interface."""
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save


def search(library, queries, library_ids, query_ids, sizes, k=8, device="cuda", chunk=128):
    if (library.ndim != 2 or queries.ndim != 2 or library.shape[1] != queries.shape[1]
            or len(library_ids) != len(library) or len(query_ids) != len(queries)
            or len(np.unique(library_ids)) != len(library_ids)
            or not np.isfinite(library).all() or not np.isfinite(queries).all()
            or any(n > len(library) or n <= k for n in sizes)):
        raise ValueError("Invalid feature arrays, identities or neighborhood sizes")
    tensor = lambda value: torch.as_tensor(value, device=device)
    x, ids = tensor(library.astype(np.float64)), tensor(library_ids)
    result = {n: dict(indices=[], distance=[]) for n in sizes}
    with torch.no_grad():
        for start in range(0, len(queries), chunk):
            q = tensor(queries[start:start+chunk].astype(np.float64))
            qi = tensor(query_ids[start:start+chunk])
            # Direct differences avoid cancellation in squared-norm MM distances.
            distance = torch.cdist(q, x, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            distance.masked_fill_(qi[:, None] == ids[None, :], float("inf"))
            for n in sizes:
                order = torch.argsort(distance[:, :n], dim=1, stable=True)[:, :k]
                result[n]["indices"].append(order.cpu().numpy())
                result[n]["distance"].append(distance.gather(1, order).cpu().numpy())
    return {n:{key:np.concatenate(parts) for key,parts in r.items()} for n,r in result.items()}


def verify_cpu(library, queries, library_ids, query_ids, result, n, sample=8):
    """Independent full-library NumPy ranking on fixed evenly spaced query rows."""
    selected = np.unique(np.linspace(0, len(queries)-1, min(sample,len(queries)), dtype=int))
    for j in selected:
        d = np.sqrt(np.square(library[:n].astype(np.float64)-queries[j].astype(np.float64)).sum(axis=1))
        d[library_ids[:n] == query_ids[j]] = np.inf
        order = np.argsort(d, kind="stable")[:result["indices"].shape[1]]
        np.testing.assert_array_equal(order, result["indices"][j])
        np.testing.assert_allclose(d[order], result["distance"][j], rtol=2e-14, atol=1e-14)
    return len(selected)


def main():
    env = configure()
    rows = arrays("/input/features.npz")
    if set(rows) != {"library_features", "query_features", "library_ids", "query_ids"}:
        raise ValueError("Neighbor worker accepts features and identities only")
    plan = read_json("/input/plan.json")
    output, timings = {}, {}
    for space, width in plan["spaces"].items():
        started = time.perf_counter()
        result = search(rows["library_features"][:, :width], rows["query_features"][:, :width],
                        rows["library_ids"], rows["query_ids"], plan["libraries"])
        torch.cuda.synchronize()
        timings[space] = time.perf_counter()-started
        for size, r in result.items():
            for key, value in r.items():
                output[f"{space}_{size}_{key}"] = value
    np.savez(Path("/output/neighbors.npz"), **output)
    save("/output/result.json", dict(status="complete", timings=timings, environment=env,
         arithmetic="FP64 direct differences on frozen FP32 features", targets_received=False))


if __name__ == "__main__":
    main()

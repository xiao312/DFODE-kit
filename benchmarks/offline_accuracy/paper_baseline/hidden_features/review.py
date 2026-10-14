"""Independent raw-array recount and source/artifact verification after fitting."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import read_json, checked_file, project_path, arrays, digest
from benchmarks.research_contract.replay import population_rows
from .prepare import save


def diversity_controls(project, run, binding):
    """Verify the changed training population against the archived 10k control."""
    control = project / "runs/offline-accuracy/hidden-features-20261013"
    selections = arrays(project / binding["transfer_root"] / "plan/selections.npz")
    records = []
    for seed in (20261011, 20261012):
        old = read_json(control / "fits" / f"{seed}-A/fit/result.json")
        new = read_json(run / "fits" / f"{seed}-S/fit/result.json")
        if old["initial_weights_sha256"] != new["initial_weights_sha256"]:
            raise ValueError("Diversity arm initialization differs from control")
        differences = {k: [old["config"].get(k), new["config"].get(k)]
                       for k in old["config"] | new["config"]
                       if old["config"].get(k) != new["config"].get(k)}
        if set(differences) != {"arm", "experiment", "training_count"}:
            raise ValueError("Unexpected diversity configuration change")
        indices = np.load(run / "prepared" / str(seed) / "training-indices.npy", allow_pickle=False)
        if len(indices) != 50000 or len(np.unique(indices)) != 50000:
            raise ValueError("Diversity training population is not 50k distinct states")
        np.testing.assert_array_equal(indices[:10000], np.arange(10000))
        if np.intersect1d(indices, selections[f"unseen_{seed}"]).size:
            raise ValueError("Candidate training contains fixed unseen evaluation rows")
        for population in ("same_source_unseen", "development"):
            relative = Path("prepared") / str(seed) / f"input-{population}/states.npz"
            before, after = arrays(control / relative), arrays(run / relative)
            for key in ("states", "source_indices"):
                np.testing.assert_array_equal(before[key], after[key])
        records.append(dict(seed=seed, initial_weights_equal=True,
                            configuration_differences=differences, training_states=len(indices),
                            fixed_unseen_overlap=0, evaluation_inputs_identical=True))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new review file")
    report = read_json(checked_file(args.run / "report.json", read_json(args.run / "verification.json")["report_sha256"]))
    if report["status"] != "complete":
        raise ValueError("Campaign not complete")
    for path, expected in report["artifacts"].items():
        checked_file(project_path(args.run, path), expected)
    binding = read_json(args.project / "research/bindings.json")
    controls = diversity_controls(args.project, args.run, binding) if any(r["arm"] == "S" for r in report["records"]) else None
    active = np.array([name != "AR" for name in binding["species"]])
    table = []
    for seed in (20261011, 20261012):
        for population, rows in population_rows(args.project / binding["prepared_root"], args.project / binding["transfer_root"], str(seed)):
            selection_path = args.run / "prepared" / str(seed) / "training-indices.npy"
            if population == "training" and selection_path.exists():
                selected = np.load(selection_path, allow_pickle=False)
                full = arrays(args.project / binding["prepared_root"] / str(seed) / "training.npz")
                rows = {k:v[selected] for k,v in full.items()}
                del full
            d, base = rows["delta"][:, active], rows["base"][:, active]
            allowance = 1e-15 + .1 * abs(d)
            old = abs(base-d) <= allowance
            for arm in sorted({r["arm"] for r in report["records"]}):
                fit = args.run / "fits" / f"{seed}-{arm}"
                pred = arrays(fit / population / "predictions.npz")
                np.testing.assert_array_equal(pred["source_indices"], rows["source_indices"])
                p = pred["prediction"][:, active]
                passed = abs(p-d) <= allowance
                expected = next(r for r in report["records"] if r["seed"] == seed and r["arm"] == arm)["populations"][population]["candidate"]
                if int(passed.sum()) != expected["groups"]["all"]["pass_count"]:
                    raise ValueError("Independent component count mismatch")
                repair, damage = int((~old & passed).sum()), int((old & ~passed).sum())
                if repair != expected["groups"]["all"]["repaired"] or damage != expected["groups"]["all"]["damaged"]:
                    raise ValueError("Independent repair/damage mismatch")
                if not np.isclose(float(passed.mean()), float(old.mean()) + (repair-damage)/old.size, rtol=0, atol=2e-16):
                    raise ValueError("Repair/damage identity failed")
                if np.count_nonzero((rows["states"][:,2:][:,active]+p)<0) != expected["endpoint_diagnostics"]["negative_components"]:
                    raise ValueError("Endpoint count mismatch")
                training = read_json(fit / "fit/result.json")
                if training["updates_completed"] != 6000 or training["row_presentations"] != 60000000:
                    raise ValueError("Scientific work mismatch")
                if arm == "A" and population != "same_source_unseen":
                    historical = args.project / binding["seeds"][str(seed)]["historical"]["coordinate-sl1"]["directory"]
                    np.testing.assert_array_equal(pred["prediction"], arrays(historical / f"{population}-predictions.npz")["prediction"])
                table.append(dict(seed=seed, arm=arm, population=population, states=len(p),
                    base_acceptance=float(old.mean()), acceptance=float(passed.mean()),
                    repaired=repair, damaged=damage, nontrivial=expected["groups"]["nontrivial"]["acceptance"],
                    p99=float(np.quantile(abs(p-d)/allowance,.99)), training_seconds=training["training_wall_seconds"],
                    parameters=training["parameter_count"]))
    save(args.output, dict(status="verified", source_report_sha256=digest(args.run / "report.json"),
                          table=table, decisions=report["decisions"],
                          historical_A_exact=True if any(r["arm"] == "A" for r in report["records"]) else None,
                          diversity_controls=controls,
                          incumbent_changed=False, confirmation_opened=False))
    print(args.output)


if __name__ == "__main__":
    main()

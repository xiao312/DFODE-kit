"""Recount real historical predictions under the new, unchanged physical scorer."""
import argparse
import json
from pathlib import Path
import numpy as np
from .adapter import arrays, digest, read_json, checked_file, checked_artifacts, project_path, aligned_score
from .gates import comparison


def inputs(project, bindings):
    root = project_path(project, bindings["prepared_root"])
    transfer = project_path(project, bindings["transfer_root"])
    plan = read_json(checked_file(transfer / "plan/plan.json", bindings["transfer_plan_sha256"]))
    checked_artifacts(transfer / "plan", plan)
    if plan["species"] != bindings["species"] or plan["dataset_manifest_sha256"] != bindings["dataset_manifest_sha256"]:
        raise ValueError("Species or reference binding differs")
    if plan["mechanism_sha256"] != bindings["mechanism_sha256"] or plan["interval_s"] != bindings["interval_s"]:
        raise ValueError("Mechanism or interval differs")
    manifests = {}
    for seed, identity in bindings["seeds"].items():
        directory = root / seed
        manifest = read_json(checked_file(directory / "manifest.json", identity["prepared_manifest_sha256"]))
        checked_artifacts(directory, manifest)
        if manifest["species"] != bindings["species"] or manifest["hashes"]["dataset_manifest"] != bindings["dataset_manifest_sha256"]:
            raise ValueError("Prepared reference binding differs")
        checked_file(directory / "base-weights.pt", identity["anchor_weights_sha256"])
        if identity["incumbent_weights_sha256"] != identity["anchor_weights_sha256"]:
            raise ValueError("M0 replay expects the unchanged initial incumbent")
        manifests[seed] = manifest
        for historical in identity["historical"].values():
            path = project_path(project, historical["directory"])
            result = read_json(checked_file(path / "result.json", historical["result_sha256"]))
            if result["status"] != "complete" or result["prepared_manifest_sha256"] != identity["prepared_manifest_sha256"]:
                raise ValueError("Historical fit is incomplete or uses another population")
            checked_artifacts(path, result)
    report = read_json(transfer / "evaluation/report.json")
    checked_file(transfer / "evaluation/report.json", read_json(transfer / "evaluation/verification.json")["report_sha256"])
    checked_artifacts(transfer / "evaluation", report)
    return root, transfer, report


def population_rows(root, transfer, seed):
    full = arrays(root / seed / "training.npz")
    selection = arrays(transfer / "plan/selections.npz")[f"unseen_{seed}"]
    yield "training", {k: v[:10000] for k, v in full.items()}
    yield "same_source_unseen", {k: v[selection] for k, v in full.items()}
    del full
    yield "development", arrays(root / seed / "development.npz")


def score(prediction, rows, species, population):
    return aligned_score(prediction, rows, species, prediction_species=species,
                         population_id=population, prediction_population_id=population)


def execute(project, bindings, policy, output, source_report):
    root, transfer = project / bindings["prepared_root"], project / bindings["transfer_root"]
    records = []
    for seed, identity in bindings["seeds"].items():
        old_report = next(v for v in source_report["seeds"] if v["seed"] == int(seed))
        for name, rows in population_rows(root, transfer, seed):
            population = f"{seed}:{name}"
            baseline = score(dict(prediction=rows["base"], source_indices=rows["source_indices"]), rows, bindings["species"], population)
            old = old_report["populations"][name]["curves"]
            if baseline["groups"]["all"]["acceptance"] != old[0]["acceptance"]:
                raise ValueError("Anchor acceptance does not reproduce")
            candidates = {}
            names = ("coordinate-sl1",) if name == "same_source_unseen" else ("direct", "physical", "coordinate-sl1")
            for method in names:
                hist = project / identity["historical"][method]["directory"]
                path = (transfer / "evaluation" / f"{seed}-{name}.npz" if name == "same_source_unseen"
                        else hist / f"{name}-predictions.npz")
                result = score(arrays(path), rows, bindings["species"], population)
                if name != "same_source_unseen":
                    expected = read_json(hist / "result.json")[name]
                    for group in ("all", "nontrivial"):
                        if result["groups"][group]["pass_count"] != expected[group]["pass_count"]:
                            raise ValueError("Historical component counts differ")
                    if result["groups"]["all"]["damaged"] != expected["pass_to_fail"]:
                        raise ValueError("Historical damage differs")
                if method == "coordinate-sl1" and result["groups"]["all"]["acceptance"] != old[-1]["acceptance"]:
                    raise ValueError("Transfer acceptance differs")
                gate = comparison(result, baseline, policy)
                if name != "training" and method in ("direct", "coordinate-sl1") and "overall_gain" not in gate["failures"]:
                    raise ValueError("Known negative candidate was not rejected")
                candidates[method] = dict(metrics=result, versus_anchor=gate, versus_incumbent=gate)
            zero = score(dict(prediction=np.zeros_like(rows["base"]), source_indices=rows["source_indices"]), rows, bindings["species"], population)
            if zero["groups"]["nontrivial"]["pass_count"] != 0:
                raise ValueError("Zero negative control passed nontrivial chemistry")
            record = dict(seed=int(seed), population=name, anchor=baseline, candidates=candidates,
                          zero_control=zero, exposure=old_report["populations"][name].get("feature_calibration_seen_states"))
            records.append(record)
            print(json.dumps(dict(seed=seed, population=name, anchor=baseline["groups"]["all"]["acceptance"],
                                  candidates={k:v["metrics"]["groups"]["all"]["acceptance"] for k,v in candidates.items()})), flush=True)
    output.mkdir(parents=True)
    report = dict(status="M0_SCORE_PARITY_PASSED", bindings=bindings, records=records,
                  source_files={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
                  limitation="Saved-array parity only; no new fit, latency qualification, isolation certification or promotion")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(dict(report_sha256=digest(output / "report.json"))), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, default=Path("research/bindings.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    bindings = read_json(args.bindings)
    campaign = read_json(args.bindings.parent / "campaign.json")
    _, _, report = inputs(args.project, bindings)
    print(json.dumps(dict(status="validated", execution=args.execute)), flush=True)
    if args.execute:
        execute(args.project, bindings, campaign["screening"], args.output, report)


if __name__ == "__main__":
    main()

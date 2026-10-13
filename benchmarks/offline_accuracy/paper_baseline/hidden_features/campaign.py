"""Trusted controller for one registered feature experiment; no automatic promotion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmarks.research_contract.adapter import read_json, digest, checked_file, arrays, verify_evaluator_lock
from benchmarks.research_contract.replay import inputs, population_rows, score
from benchmarks.research_contract.gates import comparison, decision
from benchmarks.research_contract.admission import validate, acquire_lease, process_identity
from benchmarks.research_contract.sandbox import command, probe
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.flame_conditioning.extract import source_revision
from .prepare import stage, verify_stage, save


def processes_on_gpu(gpu):
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lines = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    return {int(line.split(",")[1]) for line in lines.splitlines() if line.split(",")[0].strip() == uuid}


def descendants(parent):
    found = {parent}
    for _ in range(8):
        added = set()
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if int(fields[1]) in found:
                    added.add(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if added <= found:
            break
        found |= added
    return found


def run_worker(project, input_dir, output, args, state, gpu, model=None):
    output.mkdir(parents=True)
    if processes_on_gpu(gpu) - {os.getpid()}:
        raise RuntimeError("WAITING_RESOURCE: selected GPU has another process; nothing was stopped")
    cmd = command(project, input_dir, output, ["-m", "benchmarks.offline_accuracy.paper_baseline.hidden_features.worker", *args], model_dir=model, gpu=gpu)
    with (output / "worker.log").open("x") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={"PATH":"/usr/bin:/bin"})
        save(state, dict(status="RUNNING", pid=child.pid, start_ticks=process_identity(child.pid), output=str(output), args=args))
        started = time.monotonic()
        while child.poll() is None:
            time.sleep(2)
            foreign = processes_on_gpu(gpu) - descendants(os.getpid())
            # GPU accounting and /proc are not atomic. Our worker can exit
            # between these reads; recheck it before interpreting an orphan PID.
            if child.poll() is not None:
                break
            if foreign:
                child.terminate()  # Only our Bubblewrap child; never another user's job.
                child.wait(timeout=15)
                save(state, dict(status="WAITING_RESOURCE", output=str(output), foreign_gpu_pids=sorted(foreign),
                                 elapsed_seconds=time.monotonic()-started))
                raise RuntimeError("Resource contention: stopped our own worker; preserve partial outputs")
        elapsed = time.monotonic()-started
        save(state, dict(status="COMPLETE" if child.returncode == 0 else "FAILED", output=str(output), elapsed_seconds=elapsed))
        if child.returncode:
            raise RuntimeError(f"Worker failed; inspect {output / 'worker.log'}")
    return elapsed


def score_arm(project, output, bindings, seed, arm, campaign):
    prepared = project / bindings["prepared_root"]
    transfer = project / bindings["transfer_root"]
    result, gates, latency = {}, {}, []
    for population, rows in population_rows(prepared, transfer, str(seed)):
        selection_path = output / "prepared" / str(seed) / "training-indices.npy"
        if population == "training" and selection_path.exists():
            selected = np.load(selection_path, allow_pickle=False)
            full = arrays(prepared / str(seed) / "training.npz")
            rows = {k:v[selected] for k,v in full.items()}
            del full
        prediction_root = output / "fits" / f"{seed}-{arm}" / population
        report = read_json(prediction_root / "result.json")
        checked_file(prediction_root / "predictions.npz", report["prediction_sha256"])
        predictions = arrays(prediction_root / "predictions.npz")
        if population == "training":
            cached = arrays(prediction_root.parent / "fit/training-replay.npz")
            np.testing.assert_array_equal(cached["prediction"], predictions["prediction"])
            np.testing.assert_array_equal(cached["coordinate"], predictions["coordinate"])
        scored = score(predictions, rows, bindings["species"], f"{seed}:{population}")
        baseline = score(dict(prediction=rows["base"], source_indices=rows["source_indices"]), rows, bindings["species"], f"{seed}:{population}")
        result[population] = dict(candidate=scored, anchor=baseline)
        if population != "training":
            for comparator in ("anchor", "incumbent"):
                gates[f"{seed}:{population}:{comparator}"] = comparison(scored, baseline, campaign["screening"])
            latency.append(float(np.median(report["timings"]["candidate"]) / np.median(report["timings"]["anchor"])))
    return dict(seed=seed, arm=arm, populations=result, gates=gates, latency_ratios=latency)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment", default="EXP-20261013-hidden-features")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    evaluator_lock = verify_evaluator_lock(project)
    if args.output.exists():
        parser.error("Output already exists; inspect status/jobs instead of launching a duplicate")
    campaign = read_json(project / "research/campaign.json")
    bindings = read_json(project / "research/bindings.json")
    if args.experiment not in ("EXP-20261013-hidden-features", "EXP-20261013-preservation", "EXP-20261013-linear", "EXP-20261013-diversity"):
        raise ValueError("Use a registered experiment identifier")
    experiment = read_json(project / "research/experiments" / (args.experiment + ".json"))
    arms = [value["id"] for value in experiment["arms"]]
    inputs(project, bindings)
    m0 = project / "runs/offline-accuracy/research-contract-20261013/m0"
    checked_file(m0 / "report.json", read_json(m0 / "verification.json")["report_sha256"])
    if read_json(m0 / "report.json")["status"] != "M0_SCORE_PARITY_PASSED":
        raise ValueError("M0 missing")
    if source_revision()["dirty"]:
        raise ValueError("Commit source before execution")
    if processes_on_gpu(args.gpu):
        print(json.dumps(dict(status="WAITING_RESOURCE", gpu=args.gpu)))
        return
    print(json.dumps(dict(status="PRECHECKED", execute=args.execute, configurations=len(arms), fits=2*len(arms))), flush=True)
    if not args.execute:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output.mkdir(parents=True)
    lease = project / "runs/research-control" / (experiment["experiment_id"] + ".lease.json")
    acquire_lease(lease, dict(experiment=experiment["experiment_id"], output=str(args.output.resolve()), start_ticks=process_identity(os.getpid())))
    state = args.output / "active-job.json"
    save(args.output / "frozen-experiment.json", experiment)
    save(args.output / "source.json", source_revision())
    save(args.output / "evaluator-lock.json", evaluator_lock)
    configure()
    stage(project, args.output, bindings, arms)
    empty_input = args.output / "probe-input"
    probe_output = args.output / "probe-output"
    empty_input.mkdir(); probe_output.mkdir()
    save(args.output / "isolation.json", json.loads(probe(project, empty_input, probe_output, args.gpu)))
    readiness = dict(metric=campaign["metric"], m0_verified=True, identities_verified=True,
                     inference_isolation_verified=True, source_clean=True, idle_gpu=True)
    validate(campaign, experiment, readiness, [])
    records, elapsed = [], 0.
    for arm in arms:
        for seed in campaign["screening"]["seeds"]:
            verify_evaluator_lock(project)
            prepared = args.output / "prepared" / str(seed)
            verify_stage(prepared)
            destination = args.output / "fits" / f"{seed}-{arm}"
            elapsed += run_worker(project, prepared / ("train-" + arm), destination / "fit",
                                  ["train", "--arm", arm, "--seed", str(seed)], state, args.gpu)
            model = destination / "model"
            shutil.copytree(prepared / "model", model)
            shutil.copyfile(destination / "fit/weights.pt", model / "weights.pt")
            for population in ("training", "same_source_unseen", "development"):
                elapsed += run_worker(project, prepared / ("input-" + population), destination / population,
                                      ["infer", "--arm", arm, "--seed", str(seed)], state, args.gpu, model)
            record = score_arm(project, args.output, bindings, seed, arm, campaign)
            verify_evaluator_lock(project)
            records.append(record)
            save(destination / "score.json", record)
            print(json.dumps(dict(arm=arm, seed=seed, acceptance={p:r["candidate"]["groups"]["all"]["acceptance"] for p,r in record["populations"].items()})), flush=True)
    decisions = {}
    expected = [f"{seed}:{population}:{comparator}" for seed in campaign["screening"]["seeds"]
                for population in ("same_source_unseen", "development") for comparator in ("anchor", "incumbent")]
    for arm in arms:
        selected = [r for r in records if r["arm"] == arm]
        decisions[arm] = decision({k:v for r in selected for k,v in r["gates"].items()}, expected,
                                  max(v for r in selected for v in r["latency_ratios"]), campaign["screening"])
    statuses = {arm:value["candidate_status"] for arm,value in decisions.items()}
    save(state, dict(status="REVIEW_REQUIRED", decisions=statuses, worker_wall_seconds=elapsed))
    save(args.output / "report.json", dict(status="complete", decisions=decisions, records=records, worker_wall_seconds=elapsed,
         artifacts={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob("*") if p.is_file()}))
    save(args.output / "verification.json", dict(report_sha256=digest(args.output / "report.json")))
    print(json.dumps(dict(status="complete", decisions=statuses)), flush=True)


if __name__ == "__main__":
    main()

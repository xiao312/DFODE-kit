"""Run the fixed chemistry feasibility pilot; no training or full-grid generation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cantera as ct
import numpy as np
import scipy

from analysis import write_analysis
from chemistry import (direct_increment, endpoint, mechanism_metadata,
                       parent_trajectory, quantized_state)


def validate(config):
    if config["schema_version"] != 1 or config["cantera_version"] != "3.2.0":
        raise ValueError("Expected schema 1 and Cantera 3.2.0")
    if not 1 <= config["anchors_per_trajectory"] <= 8:
        raise ValueError("Pilot supports one to eight anchors")
    if not 1 <= config["wall_time_seconds"] <= 3600:
        raise ValueError("Pilot wall-time limit must be at most 3600 seconds")
    for key in ("pressure_Pa", "equivalence_ratio"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("temperatures_K", "intervals_s"):
        if not config[key] or not np.all(np.isfinite(config[key])) or min(config[key]) <= 0:
            raise ValueError(f"Invalid {key}")
    expected = ["default", "relative10", "relative12", "absolute18", "absolute21"]
    if [item["id"] for item in config["tolerances"]] != expected:
        raise ValueError("Tolerance identifiers/order do not match the analysis contract")
    if [item["id"] for item in config["radau_tolerances"]] != ["radau_check", "radau_reference"]:
        raise ValueError("Both independent integration checks are required")
    for item in config["tolerances"] + config["radau_tolerances"]:
        if any(not np.isfinite(item[key]) or item[key] <= 0 for key in ("rtol", "atol")):
            raise ValueError("Tolerances must be positive and finite")
    for mechanism in config["mechanisms"]:
        if mechanism["id"] not in ("h2", "ch4") or mechanism["duration_s"] <= 0:
            raise ValueError("Pilot supports only H2 and CH4 with positive duration")
    for key in ("species_budget", "temperature_budget"):
        if any(not np.isfinite(config[key][name]) or config[key][name] <= 0 for name in ("atol", "rtol")):
            raise ValueError("Physical budgets must be positive and finite")


def revision():
    root = Path(__file__).resolve().parents[3]
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_interval(mechanism, state, interval, config, deadline):
    solutions, failures = {}, {}
    def attempt(name, operation):
        if time.monotonic() > deadline:
            raise TimeoutError("Pilot wall-time limit reached")
        try:
            result = operation()
            if not np.all(np.isfinite(result["delta"])):
                raise ValueError("Non-finite increment")
            solutions[name] = result
        except TimeoutError:
            raise
        except Exception as error:
            failures[name] = str(error)[:2000]
    for tolerance in config["tolerances"]:
        attempt(tolerance["id"], lambda tolerance=tolerance: endpoint(
            mechanism, state, interval, tolerance["rtol"], tolerance["atol"]))
    tight = config["tolerances"][-1]
    attempt("step_limited", lambda: endpoint(mechanism, state, interval,
                                            tight["rtol"], tight["atol"], interval / 10))
    for tolerance in config["radau_tolerances"]:
        attempt(tolerance["id"], lambda tolerance=tolerance: direct_increment(
            mechanism, state, interval, config, tolerance["rtol"], tolerance["atol"], deadline))
    rounded, quantization = quantized_state(state)
    attempt("fp32_input", lambda: endpoint(mechanism, rounded, interval, tight["rtol"], tight["atol"]))
    return {"solutions": solutions, "failures": failures,
            "quantized_state": rounded, "quantization": quantization}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/reference-pilot/run-001"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-records", type=int, help="Explicit smoke-test limit; never used for a complete pilot")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    if ct.__version__ != config["cantera_version"]:
        raise RuntimeError(f"Expected Cantera {config['cantera_version']}, found {ct.__version__}")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    count = len(config["mechanisms"]) * len(config["temperatures_K"]) * config["anchors_per_trajectory"] * len(config["intervals_s"])
    if args.dry_run:
        print(json.dumps({"planned_intervals": count, "solves_per_interval": 9,
                          "output": str(args.output), "max_records": args.max_records}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "source": revision(), "host": os.environ.get("DFODE_RUN_HOST", platform.node()),
                "container_hostname": platform.node(), "platform": platform.platform(),
                "container_image": os.environ.get("DFODE_CONTAINER_IMAGE"),
                "versions": {"python": platform.python_version(), "cantera": ct.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "sundials": getattr(ct, "__sundials_version__", "not exposed by wheel")},
                "installed_packages": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
                "thread_limits": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "mechanisms": [mechanism_metadata(item) for item in config["mechanisms"]],
                "planned_intervals": count, "max_records": args.max_records,
                "completed_intervals": 0, "status": "running", "trajectory_failures": []}
    manifest_path = args.output / "manifest.json"
    def save():
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    save()
    try:
        with (args.output / "intervals.jsonl").open("w", buffering=1) as stream:
            for mechanism in config["mechanisms"]:
                for temperature in config["temperatures_K"]:
                    trajectory_id = f"{mechanism['id']}-{temperature:g}K"
                    try:
                        trajectory = parent_trajectory(mechanism, temperature, config)
                    except Exception as error:
                        manifest["trajectory_failures"].append({"id": trajectory_id, "error": str(error)[:2000]})
                        save()
                        continue
                    (args.output / f"trajectory-{trajectory_id}.json").write_text(json.dumps(trajectory, allow_nan=False))
                    for anchor_index, state in enumerate(trajectory["anchors"]):
                        for interval in config["intervals_s"]:
                            if args.max_records and manifest["completed_intervals"] >= args.max_records:
                                manifest["status"] = "smoke_limit"
                                return
                            record_id = f"{trajectory_id}-a{anchor_index}-h{interval:g}"
                            record = {"id": record_id, "trajectory_id": trajectory_id,
                                      "mechanism": mechanism["id"], "state": state, "interval_s": interval,
                                      **run_interval(mechanism, state, interval, config, started + config["wall_time_seconds"])}
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            manifest["completed_intervals"] += 1
                            save()
                            print(json.dumps({"completed": manifest["completed_intervals"], "id": record_id,
                                              "failures": len(record["failures"])}), flush=True)
        manifest["status"] = "complete" if manifest["completed_intervals"] == count else "incomplete"
    except TimeoutError:
        manifest["status"] = "time_limit"
    finally:
        save()
        write_analysis(args.output)


if __name__ == "__main__":
    main()

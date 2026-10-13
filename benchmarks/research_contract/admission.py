"""Run admission is separate from candidate quality and never changes the grader."""
import json
import os
from pathlib import Path


def validate(campaign, experiment, readiness, active_jobs):
    if not campaign["execution_authorized"]:
        raise ValueError("Owner execution authorization required")
    if campaign["metric"] != readiness["metric"]:
        raise ValueError("Physical contract differs from pinned evaluator")
    for key in ("m0_verified", "identities_verified", "inference_isolation_verified", "source_clean"):
        if readiness.get(key) is not True:
            raise ValueError("Admission blocked: " + key)
    if active_jobs:
        raise ValueError("Reconcile active job before starting another fit")
    if readiness.get("idle_gpu") is not True:
        return "WAITING_RESOURCE"
    if experiment["budget"]["updates"] <= 0 or experiment["budget"]["batch_size"] <= 0:
        raise ValueError("Declare finite scientific work per fit")
    return "READY"


def acquire_lease(path, record):
    """Exclusive create prevents duplicate controller launches; never steal a lease."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record, controller_pid=os.getpid())
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, allow_nan=False)
    return record


def process_identity(pid):
    """Linux start ticks detect PID reuse after reconnect. None means no such PID."""
    try:
        value = Path(f"/proc/{int(pid)}/stat").read_text()
    except FileNotFoundError:
        return None
    # Command names can contain spaces and parentheses; starttime is field 22.
    return value.rsplit(")", 1)[1].split()[19]

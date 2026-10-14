"""Pure row checks and bounded spawn workers, each with a private reactor."""
import multiprocessing as mp
import time

import numpy as np

from benchmarks.flame_conditioning.chemistry import EndpointIntegrator

_integrator = None
_interval = None


def initialize(mechanism, config):
    global _integrator, _interval
    _integrator = EndpointIntegrator(mechanism, config["cvode_rtol"], config["cvode_atol"])
    _interval = config["interval_s"]


def label_rows(integrator, states, interval):
    """Same exclusion rules as prepare.py; exceptions retain an explicit row error."""
    delta = np.full((len(states), states.shape[1]-2), np.nan)
    accepted = np.zeros(len(states), dtype=bool)
    records = []
    for index, row in enumerate(states):
        state = dict(T=float(row[0]), P=float(row[1]), Y=row[2:].tolist())
        record = dict(row=index)
        try:
            result = integrator.advance(state, interval)
            delta[index] = result.pop("delta")
            record.update(result)
            physical = result["diagnostics"]
            if not np.isfinite(delta[index]).all() or physical["minimum_mass_fraction"] < 0:
                raise ValueError("Nonfinite label or negative endpoint; retained but excluded")
            if abs(physical["mass_delta_sum"]) > 1e-10 or physical["element_delta_max"] > 1e-10:
                raise ValueError("Label conservation failure")
            if abs(physical["temperature_change_K"]) > 1e-8 or abs(physical["relative_density_change"]) > 1e-10:
                raise ValueError("Chemistry constraint failure")
            accepted[index] = True
        except Exception as error:
            record["error"] = str(error)
        records.append(record)
    return delta, accepted, records


def task(arguments):
    start, states = arguments
    delta, accepted, records = label_rows(_integrator, states, _interval)
    for record in records:
        record["row"] += start
    return start, delta, accepted, records


def chunks(mechanism, config, states, ranges, workers, deadline):
    """Yield bounded results. Timeout terminates only this runner's worker pool."""
    if not 1 <= workers <= 8:
        raise ValueError("Use 1 to 8 allocated workers")
    if not ranges:
        return
    with mp.get_context("spawn").Pool(workers, initialize, (str(mechanism), config)) as pool:
        # Queue at most two chunks per worker; avoid pickling the full dataset.
        for offset in range(0, len(ranges), 2*workers):
            batch = ranges[offset:offset+2*workers]
            jobs = ((start, states[start:stop]) for start, stop in batch)
            pending = pool.imap_unordered(task, jobs, chunksize=1)
            for _ in batch:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Reference generation reached its wall limit")
                try:
                    yield pending.next(timeout=remaining)
                except mp.TimeoutError as error:
                    raise TimeoutError("Reference generation reached its wall limit") from error

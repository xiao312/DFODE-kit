"""One finite reference calculation, no training or historical-label writes."""
import argparse
import json
from pathlib import Path
import time
import cantera as ct
import numpy as np
from benchmarks.flame_conditioning.chemistry import set_state
from ..correction_transfer.audit_worker import reference_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('mechanism', type=Path); p.add_argument('index', type=int)
    a = p.parse_args()
    with np.load(a.root/'states.npz') as data: row = data['states'][a.index]
    state = dict(T=float(row[0]), P=float(row[1]), Y=row[2:].tolist())
    gas = ct.Solution(str(a.mechanism)); set_state(gas, state)
    effective = np.r_[gas.T, gas.P, gas.Y]
    start = time.monotonic(); checks, failures = reference_record(a.mechanism, state, 1e-6)
    print(json.dumps(dict(index=a.index, status='incomplete' if failures else 'complete',
                         effective_state=effective.tolist(), checks=checks, failures=failures,
                         seconds=time.monotonic()-start), allow_nan=False))


if __name__ == '__main__': main()

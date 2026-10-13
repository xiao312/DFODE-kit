"""One bounded CPU reference state; invoked in a killable subprocess."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from benchmarks.flame_conditioning.chemistry import endpoint,direct_increment


def reference_record(mechanism,state,interval):
    deadline = time.monotonic()+110
    checks,failures = {},[]
    for name,atol,max_step in (("tight",1e-18,None),("tighter",1e-21,None),("step-limited",1e-21,interval/10)):
        try:
            checks[name] = endpoint(mechanism,state,interval,1e-12,atol,max_step)
        except Exception as error:
            failures.append(dict(check=name,error=str(error)))
    for name,rtol,atol in (("direct",1e-9,1e-5),("direct-tighter",1e-11,1e-7)):
        try:
            checks[name] = direct_increment(mechanism,state,interval,rtol,atol,deadline)
        except Exception as error:
            failures.append(dict(check=name,error=str(error)))
    return checks,failures


def reference_scores(state,ordinary,base,checks,active,anchor):
    reference = np.asarray(checks["direct-tighter"]["delta"])
    deltas = np.asarray([v["delta"] for v in checks.values()])
    initial = np.asarray(state["Y"])
    uncertainty = np.maximum.reduce([abs(deltas-reference).max(axis=0),
        abs(np.spacing(initial)),abs(np.spacing(initial+reference))])
    allowance = 1e-15+.1*abs(reference)
    discrepancy = abs(ordinary-reference)/allowance
    qualified = uncertainty<=.1*allowance
    stable = (np.sign(deltas)==np.sign(reference)).all(axis=0)
    sign_match = np.sign(ordinary)==np.sign(reference)
    return dict(reference_delta=reference.tolist(),uncertainty=uncertainty.tolist(),
        nominal_difference_in_allowances=discrepancy.tolist(),qualified=qualified.tolist(),
        sign_stable=stable.tolist(),nominal_sign_match=sign_match.tolist(),
        active_components=int(active.sum()),qualified_active=int(qualified[active].sum()),
        nominal_over_tenth=int((discrepancy[active]>.1).sum()),nominal_over_one=int((discrepancy[active]>1).sum()),
        max_nominal_difference=float(discrepancy[active].max()),
        anchor=dict(reference=float(reference[anchor]),nominal=float(ordinary[anchor]),base=float(base[anchor]),
            qualified=bool(qualified[anchor]),uncertainty_in_allowances=float(uncertainty[anchor]/allowance[anchor]),
            nominal_difference_in_allowances=float(discrepancy[anchor]),
            sign_stable=bool(stable[anchor]),nominal_sign_match=bool(sign_match[anchor]),
            base_sign_reversal=bool(np.sign(base[anchor])*np.sign(reference[anchor])<0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path); p.add_argument("mechanism",type=Path); p.add_argument("index",type=int)
    args = p.parse_args()
    plan = json.loads((args.plan/"plan.json").read_text())
    selected = json.loads((args.plan/"audit-selection.json").read_text())[args.index]
    with np.load(args.plan/"selections.npz") as data:
        row,ordinary,base = (data[name][args.index] for name in ("audit_states","audit_ordinary","audit_base"))
    state = dict(T=float(row[0]),P=float(row[1]),Y=row[2:].tolist())
    started = time.monotonic()
    checks,failures = reference_record(args.mechanism,state,plan["interval_s"])
    active = np.array([s!="AR" for s in plan["species"]])
    scores = {} if failures else reference_scores(state,ordinary,base,checks,active,plan["species"].index(selected["species"]))
    print(json.dumps(dict(index=args.index,selection=selected,state=state,checks=checks,scores=scores,
                         failures=failures,status="incomplete" if failures else "complete",
                         seconds=time.monotonic()-started),allow_nan=False))


if __name__=="__main__":
    main()

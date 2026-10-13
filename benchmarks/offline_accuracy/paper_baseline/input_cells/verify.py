"""Separate inverse-boundary verification without another reference solve."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest
from benchmarks.offline_accuracy.paired.coordinates import decode
from ..hidden_features.prepare import save
from .math import output_interval


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--project', type=Path, required=True)
    root = p.parse_args().project/'runs/offline-accuracy/input-cells-20261013'
    target = root/'independent-boundaries.json'
    if target.exists(): raise ValueError('Verification exists')
    report = read_json(root/'report.json'); review = read_json(root/'review.json')
    checked_file(root/'report.json', review['report_sha256'])
    for name, sha in report['artifacts'].items(): checked_file(root/name, sha)
    selection = read_json(root/'selection.json'); states = arrays(root/'states.npz')['states']
    largest_violation = 0.; components = 0
    for pair in selection['pairs']:
        prep = arrays(root/f"preprocessing-{pair['seed']}.npz"); active = prep['active']
        for i in pair['indices']:
            record = read_json(root/f'record-{i:03d}.json')
            if record['status'] != 'complete': continue
            # Full FP64 composition identity, not merely equal rounded inputs.
            np.testing.assert_array_equal(states[i,2:], np.array(record['effective_state'])[2:])
            y = states[i,2:][active]
            d = np.array(record['checks']['direct-tighter']['delta'])[active]
            all_d = np.array([r['delta'] for r in record['checks'].values()])[:,active]
            u = np.max(np.vstack([abs(all_d-d), abs(np.spacing(y)), abs(np.spacing(y+d))]), axis=0)
            lo, hi, valid = output_interval(y, d, u, prep)
            if not valid.all(): raise ValueError('Invalid physical reference interval')
            lower_physical = np.maximum(-y, (d-u)-1e-15-.1*abs(d-u))
            upper_physical = (d+u)+1e-15+.1*abs(d+u)
            finite = np.isfinite(lo)
            low, _ = decode(y[finite], lo[finite]*prep['y_scale'][finite]+prep['y_offset'][finite], 'gbct')
            high, _ = decode(y, hi*prep['y_scale']+prep['y_offset'], 'gbct')
            allowance = 1e-15+.1*abs(d)
            violation = max(0., np.max((low-lower_physical[finite])/allowance[finite], initial=0.),
                            np.max((upper_physical-high)/allowance, initial=0.))
            largest_violation = max(largest_violation, float(violation)); components += len(d)
            assert (lower_physical[~finite] == -y[~finite]).all()
    largest_witness = max(r.get('overlap_witness_violation_allowances',0.) for r in review['records'])
    passed = largest_violation <= 1e-6 and largest_witness <= 1e-6
    save(target, dict(status='verified' if passed else 'inconclusive_numerics', review_sha256=digest(root/'review.json'),
                      boundary_components=components, exact_effective_composition=True,
                      maximum_boundary_violation_allowances=largest_violation,
                      maximum_overlap_witness_violation_allowances=largest_witness, numerical_margin_allowances=1e-6))
    print(digest(target)); print(dict(passed=passed, components=components, boundary_violation=largest_violation))
    if not passed: raise ValueError('Do not interpret incompatible intervals')


if __name__ == '__main__': main()

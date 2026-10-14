"""Reference-qualified interval test; no model selection or label changes."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest
from benchmarks.offline_accuracy.paired.coordinates import decode
from ..hidden_features.prepare import save
from .math import neural_input, output_interval


def reference(record, initial):
    if record['status'] != 'complete' or len(record['checks']) != 5:
        return None
    d = np.asarray(record['checks']['direct-tighter']['delta'])
    all_d = np.asarray([r['delta'] for r in record['checks'].values()])
    u = np.maximum.reduce([abs(all_d-d).max(axis=0), abs(np.spacing(initial)), abs(np.spacing(initial+d))])
    if not np.isfinite([d,u]).all(): raise ValueError('Nonfinite reference')
    return d, u


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--project', type=Path, required=True)
    project = p.parse_args().project; root = project/'runs/offline-accuracy/input-cells-20261013'
    if (root/'review.json').exists(): raise ValueError('Review exists')
    report = read_json(root/'report.json')
    for name, sha in report['artifacts'].items(): checked_file(root/name, sha)
    selection = read_json(root/'selection.json'); states = arrays(root/'states.npz')['states']
    active = np.array([s != 'AR' for s in selection['species']]); results = []; saved = {}
    for k, pair in enumerate(selection['pairs']):
        prep = arrays(root/f"preprocessing-{pair['seed']}.npz")
        np.testing.assert_array_equal(prep['active'], active)
        indices = pair['indices']; rows = states[indices]
        records = [read_json(root/f'record-{i:03d}.json') for i in indices]
        if not all(r['status'] == 'complete' for r in records):
            results.append(dict(pair=k, status='reference_incomplete')); continue
        nominal_bits = [neural_input(row, prep).view('uint32') for row in rows]
        effective_bits = [neural_input(np.asarray(r['effective_state']), prep).view('uint32') for r in records]
        bits_ok = all(np.array_equal(nominal_bits[0], bits) for bits in nominal_bits+effective_bits)
        refs = [reference(r, row[2:]) for r,row in zip(records,rows)]
        intervals = []; qualified = []; differences = []
        for row, (d,u) in zip(rows,refs):
            intervals.append(output_interval(row[2:][active], d[active], u[active], prep))
            qualified.append(u[active] <= .1*(1e-15+.1*abs(d[active])))
        lo = np.maximum(intervals[0][0], intervals[1][0]); hi = np.minimum(intervals[0][1], intervals[1][1])
        eligible = qualified[0] & qualified[1] & intervals[0][2] & intervals[1][2] & bits_ok
        disjoint = lo > hi
        # Independent inverse check on an intersection witness. This does not
        # use the forward interval helper; it checks both physical constraints.
        witness = np.where(np.isfinite(lo), lo/2+hi/2, hi-1)
        overlap = eligible & ~disjoint
        for row,(d,u) in zip(rows,refs):
            physical,_ = decode(row[2:][active], witness*prep['y_scale']+prep['y_offset'], 'gbct')
            low = d[active]-u[active]-1e-15-.1*abs(d[active]-u[active])
            high = d[active]+u[active]+1e-15+.1*abs(d[active]+u[active])
            violation = np.maximum(low-physical, physical-high)/(1e-15+.1*abs(d[active]))
            differences.append(float(max(0., np.max(violation[overlap], initial=0.))))
        change = abs(refs[1][0]-refs[0][0])/(1e-15+.1*abs(refs[0][0]))
        results.append(dict(pair=k, seed=pair['seed'], source_row=pair['source_row'], status='checked',
                            exact_input_collision=bits_ok, qualified_components=int(eligible.sum()),
                            disjoint_components=int((eligible & disjoint).sum()),
                            disjoint_species=[s for s,flag in zip(np.array(selection['species'])[active],eligible & disjoint) if flag],
                            max_increment_change_allowances=float(change[active].max()),
                            overlap_witness_violation_allowances=max(differences),
                            max_effective_state_difference=float(max(np.max(abs(np.asarray(r['effective_state'])-row)) for r,row in zip(records,rows)))))
        saved[f'{k}-lower'] = lo; saved[f'{k}-upper'] = hi; saved[f'{k}-eligible'] = eligible
        saved[f'{k}-references'] = np.array(refs)
    np.savez(root/'intervals.npz', **saved)
    save(root/'review.json', dict(status='verified_diagnostic', report_sha256=digest(root/'report.json'),
         intervals_sha256=digest(root/'intervals.npz'), records=results,
         qualified_components=sum(r.get('qualified_components',0) for r in results),
         incompatible_components=sum(r.get('disjoint_components',0) for r in results),
         limitations='Constructed local training-domain pairs only. Empirical reference uncertainty, not certified bounds. No fitted model or overall precision ceiling. FP64 initial composition remains part of decoder.'))
    print(digest(root/'review.json')); print(results)


if __name__ == '__main__': main()

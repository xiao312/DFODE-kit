"""Independent categorical overlap and centred-dot correlation recount."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest
from ..hidden_features.prepare import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    project = p.parse_args().project
    root = project/'runs/offline-accuracy/representation-pairs-20261013'
    review = read_json(root/'review.json')
    for name, sha in review['artifacts'].items():
        checked_file(root/name, sha)
    output = root/'independent-overlap.json'
    if output.exists():
        raise ValueError('Check already exists')
    plan = read_json(root/'frozen-experiment.json')
    predictions = arrays(root/'predictions.npz')
    active = np.array([s != 'AR' for s in read_json(project/'research/bindings.json')['species']])
    records = []
    for seed in plan['seeds']:
        other = next(s for s in plan['seeds'] if s != seed)
        row = arrays(project/plan['source_run']/'prepared'/str(seed)/'training-reference.npz')
        np.testing.assert_array_equal(row['source_indices'], predictions['source_indices'])
        d = row['delta'][:, active]
        b = predictions[f'{other}-gbct-other-seed-member'][:, active]
        for item in (r for r in review['overlaps'] if r['seed'] == seed):
            target = item['target']; q = predictions[f'{seed}-{target}-member'][:, active]
            for group, reported in item['groups'].items():
                mask = {'all': np.ones(d.shape, bool), 'floor': abs(d) <= 1e-15/.9,
                        'weak': (abs(d) > 1e-15/.9) & (abs(d) <= 1e-12),
                        'active': abs(d) > 1e-12}[group]
                truth, first, second = d[mask], b[mask], q[mask]
                tau = 1e-15+.1*abs(truth)
                passed = 2*(abs(first-truth) <= tau).astype(int)+(abs(second-truth) <= tau)
                counts = np.bincount(passed, minlength=4)
                for key, value in dict(components=len(truth), both_fail=counts[0], second_only=counts[1],
                                       first_only=counts[2], both_pass=counts[3], oracle_union_pass=counts[1:].sum()).items():
                    assert reported[key] == int(value), (seed, target, group, key)
                sign1, sign2 = np.sign(first-truth), np.sign(second-truth)
                assert reported['opposite_error'] == int((sign1*sign2 < 0).sum())
                assert reported['both_fail_opposite'] == int(((passed == 0) & (sign1 != sign2)).sum())
                e = np.arcsinh(first/1e-14)-np.arcsinh(truth/1e-14)
                f = np.arcsinh(second/1e-14)-np.arcsinh(truth/1e-14)
                e -= e.mean(); f -= f.mean()
                norm = np.sqrt(np.dot(e, e)*np.dot(f, f))
                corr = None if norm == 0 else float(np.dot(e, f)/norm)
                if corr is None: assert reported['correlation'] is None
                else: np.testing.assert_allclose(corr, reported['correlation'], atol=1e-12, rtol=0)
            member = next(r for r in review['summary'] if r['seed'] == seed and r['target'] == target and r['rule'] == 'member')
            for rule in ('mean', 'strict'):
                candidate = next(r for r in review['summary'] if r['seed'] == seed and r['target'] == target and r['rule'] == rule)
                records.append(dict(seed=seed, target=target, rule=rule,
                                    gain_over_better_member=candidate['acceptance']-max(member['acceptance'], candidate['matched_base'])))
    save(output, dict(status='verified', review_sha256=digest(root/'review.json'), records=records,
                      limit='Seed-associated diversity includes different training/calibration samples, not initialization alone. Cached training predictions only.'))
    print(digest(output))


if __name__ == '__main__':
    main()

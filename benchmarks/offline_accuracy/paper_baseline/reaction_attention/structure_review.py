"""Read-only raw-encoding ranks and paired effects; no candidate selection."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest
from benchmarks.flame_conditioning.extract import source_revision


def spectrum(matrix, covariance=None):
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.
    strict = largest * 1e-12
    coarse = largest * max(matrix.shape) * np.finfo(np.float32).eps
    result = dict(input_dimensions=matrix.shape[0], channels=matrix.shape[1],
                rank_fp64_threshold=int((singular > strict).sum()),
                rank_fp32_threshold=int((singular > coarse).sum()),
                singular_values=singular.tolist(), thresholds=[strict, coarse])
    if covariance is not None:
        _, _, directions = np.linalg.svd(matrix.T, full_matrices=matrix.shape[1]<matrix.shape[0])
        null = directions[result['rank_fp64_threshold']:]
        total_variance = float(np.trace(covariance))
        null_variance = float(np.trace(null@covariance@null.T))
        result.update(training_variance_in_null_directions=null_variance,
            training_total_variance=total_variance,
            training_null_variance_fraction=null_variance/total_variance if total_variance>0 else None)
    return result


def structural_maps(graph, species, ordinary=None):
    matrices = []
    for name in ('reactant', 'product', 'collider'):
        value = graph[name].astype(float).copy()
        total = value.sum(0)
        value[:, total > 0] /= total[total > 0]
        matrices.append(value)
    active = graph['active']
    ycov = bcov = None
    if ordinary is not None:
        ycov = np.cov(ordinary[:, 2:2+len(species)].astype(float), rowvar=False)
        bcov = np.cov(ordinary[:, 2+len(species):].astype(float), rowvar=False)
    records = []
    for i in np.flatnonzero(active):
        support = graph['pooling'][i] > 0
        own = np.eye(len(species))[:, i:i+1]
        available = np.column_stack([m[:, support] for m in matrices] + [own])
        records.append(dict(species=species[i], reactions=int(support.sum()),
            prepared_y=spectrum(available, ycov), base_output=spectrum(available[active], bcov)))
    global_map = np.column_stack(matrices)
    return dict(global_prepared_y=spectrum(global_map, ycov),
                global_base_output=spectrum(global_map[active], bcov), per_species=records)


def main():
    from ..hidden_features.prepare import save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args(); project = args.project
    manifest = project/'research/experiments/EXP-20261014-reaction-information.json'
    plan = read_json(manifest); root = project/plan['parent']
    review = read_json(checked_file(root/'review.json', plan['parent_review_sha256']))
    report = read_json(checked_file(root/'report.json', review['report_sha256']))
    scores = read_json(checked_file(root/'protected-scores.json', review['scores_sha256']))
    binding = read_json(project/'research/bindings.json')
    frozen = read_json(root/'frozen-experiment.json')
    output = project/plan['output']
    if output.exists():
        raise ValueError('Existing diagnostic: reconcile instead of overwriting')
    if source_revision()['dirty']:
        raise ValueError('Commit diagnostic before execution')
    fits = []; structures = []
    for seed in frozen['seeds']:
        for name in ('graph.npz', 'result.json', 'feature-audit.npz'):
            checked_file(root/str(seed)/name, report['artifacts'][f'{seed}/{name}'])
        graph = arrays(root/str(seed)/'graph.npz')
        audit = arrays(root/str(seed)/'feature-audit.npz')
        structure = structural_maps(graph, binding['species'], audit['ordinary'])
        structures.append(dict(seed=seed, **structure))
        if seed != frozen['seeds'][0]:
            for key, value in arrays(root/str(frozen['seeds'][0])/'graph.npz').items():
                np.testing.assert_array_equal(graph[key], value)
        for record in read_json(root/str(seed)/'result.json')['records']:
            arm = record['arm']
            config = read_json(checked_file(root/str(seed)/arm/'config.json',
                               report['artifacts'][f'{seed}/{arm}/config.json']))
            for key in ('seed', 'width', 'heads', 'batch_size', 'learning_rate', 'final_learning_rate'):
                expected = seed if key == 'seed' else frozen[key]
                if config[key] != expected:
                    raise ValueError(f'Configuration mismatch: {key}')
            expected = 20449 if arm.startswith('pool-') else 22561
            if record['parameter_count'] != expected:
                raise ValueError('Parameter count differs')
            fits.append(dict(seed=seed, arm=arm, parameter_count=expected,
                training_wall_seconds=record['training_wall_seconds'], history=record['history']))
    contrasts = []
    for seed in frozen['seeds']:
        for population in ('training', 'same_source_unseen', 'development'):
            selected = {r['arm']:r['candidate']['groups'] for r in scores['records']
                        if r['seed'] == seed and r['population'] == population}
            for left, right in [('pool-state','pool-affinity'), ('attention-state','attention-affinity'),
                                ('pool-state','attention-state'), ('pool-affinity','attention-affinity')]:
                contrasts.append(dict(seed=seed, population=population, control=left, candidate=right,
                    acceptance_gain_pp={g:100*(selected[right][g]['acceptance']-selected[left][g]['acceptance'])
                                        for g in selected[left]},
                    additional_repairs=selected[right]['all']['repaired']-selected[left]['all']['repaired'],
                    additional_damage=selected[right]['all']['damaged']-selected[left]['all']['damaged']))
    output.mkdir(parents=True)
    save(output/'review.json', dict(status='VERIFIED_DIAGNOSTIC', source=source_revision(),
        plan_sha256=digest(manifest), parent_review_sha256=digest(root/'review.json'),
        structures=structures, fits=fits, contrasts=contrasts, candidate_promoted=False,
        limits='Raw linear encoding rank is not learned invertibility or evidence of physically relevant ambiguity. Affinity is an additional nonlinear feature not assessed by state-map rank. T/P remain direct inputs. Saved development contrasts are descriptive, not independent confirmation.'))
    print('VERIFIED_DIAGNOSTIC', digest(output/'review.json'))
    print('Global prepared-Y rank:', structure['global_prepared_y']['rank_fp64_threshold'])
    print('Per-query prepared-Y ranks:', sorted(set(r['prepared_y']['rank_fp64_threshold'] for r in structure['per_species'])))


if __name__ == '__main__':
    main()

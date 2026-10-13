"""Prepare exact rounding-cell pairs and run bounded reference subprocesses."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest, verify_evaluator_lock
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from .math import collision


def run_one(root, mechanism, index, deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 0: return dict(index=index, status='not_started')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    try:
        proc = subprocess.run([sys.executable, '-m', 'benchmarks.offline_accuracy.paper_baseline.input_cells.worker',
                               str(root), str(mechanism), str(index)], env=env, timeout=min(120, remaining), text=True, capture_output=True)
        if proc.returncode: return dict(index=index, status='failed', error=proc.stderr[-4000:])
        result = json.loads(proc.stdout); result['stderr'] = proc.stderr
        return result
    except subprocess.TimeoutExpired:
        return dict(index=index, status='timeout')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); project = a.project.resolve()
    plan = read_json(project/'research/experiments/EXP-20261013-input-cells.json')
    binding = read_json(project/'research/bindings.json'); verify_evaluator_lock(project)
    root = project/plan['run']; source = project/plan['source_run']
    if root.exists(): raise ValueError('Existing diagnostic; inspect records, do not duplicate')
    previous = read_json(checked_file(source/'review.json', plan['source_review_sha256']))
    report = read_json(checked_file(source/'report.json', previous['report_sha256']))
    for name, sha in report['artifacts'].items(): checked_file(source/name, sha)
    mechanism = checked_file(project/'runs/offline-accuracy/fuel-recipe-200k-20261009/dataset/mechanism.yaml', binding['mechanism_sha256'])
    pairs, states, preparations, skipped = [], [], {}, []
    for seed in plan['seeds']:
        directory = project/binding['prepared_root']/str(seed)
        manifest = read_json(checked_file(directory/'manifest.json', binding['seeds'][str(seed)]['prepared_manifest_sha256']))
        prep_path = checked_file(directory/'preprocessing.npz', manifest['artifacts']['preprocessing.npz'])
        raw = arrays(prep_path); prep = {k[5:]: v for k,v in raw.items() if k.startswith('base_')}
        preparations[seed] = prep
        rows = arrays(source/'prepared'/str(seed)/'training-reference.npz')
        snapshots = np.unique(rows['snapshot'])
        if len(snapshots) != 4: raise ValueError('Expected four bound snapshots')
        for snapshot in snapshots:
            for i in np.flatnonzero(rows['snapshot'] == snapshot)[:plan['rows_per_snapshot']]:
                original = rows['states'][i]
                other, meta = collision(original, prep, binding['species'].index('N2'), [binding['species'].index('AR')])
                item = dict(seed=seed, source_row=int(rows['source_indices'][i]), snapshot=float(snapshot), **meta)
                if other is None: skipped.append(item); continue
                item['indices'] = [len(states), len(states)+1]
                states.extend([original, other]); pairs.append(item)
    if len(states) > plan['resources']['maximum_states']: raise ValueError('State budget exceeded')
    print(json.dumps(dict(status='prechecked', pairs=len(pairs), skipped=len(skipped))), flush=True)
    if not a.execute: return
    revision = source_revision()
    if revision['dirty']: raise ValueError('Commit source before reference execution')
    root.mkdir(parents=True); save(root/'frozen-experiment.json', plan); save(root/'source.json', revision)
    np.savez(root/'states.npz', states=np.asarray(states))
    for seed, prep in preparations.items(): np.savez(root/f'preprocessing-{seed}.npz', **prep)
    save(root/'selection.json', dict(pairs=pairs, skipped=skipped, species=binding['species'], mechanism_sha256=binding['mechanism_sha256']))
    ticks = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
    save(root/'active-job.json', dict(pid=os.getpid(), start_ticks=ticks, source=revision, state='running', cpu_workers=4))
    started = time.monotonic(); records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(run_one, root, mechanism, i, started+plan['resources']['maximum_seconds']) for i in range(len(states))]
        for future in as_completed(tasks):
            record = future.result(); save(root/f"record-{record['index']:03d}.json", record)
            records.append({k:v for k,v in record.items() if k not in ('checks','stderr','effective_state')})
            save(root/'progress.json', dict(completed=len(records), total=len(states), records=records, seconds=time.monotonic()-started))
            print(json.dumps(dict(completed=len(records), total=len(states), status=record['status'])), flush=True)
    save(root/'report.json', dict(status='complete' if all(r['status']=='complete' for r in records) else 'incomplete',
                                 records=records, elapsed_seconds=time.monotonic()-started,
                                 artifacts={f.name:digest(f) for f in root.iterdir() if f.is_file() and f.name not in ('active-job.json','progress.json')}))
    save(root/'active-job.json', dict(pid=os.getpid(), start_ticks=ticks, source=revision, state='finished', cpu_workers=4))
    print('REFERENCE_COMPLETE', flush=True)


if __name__ == '__main__': main()

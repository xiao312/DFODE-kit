"""One registered preflight-only infrastructure retry; never steal an old lease."""
import hashlib
from benchmarks.research_contract.adapter import read_json
from benchmarks.research_contract.admission import process_identity


def validate_retry(project, output, experiment):
    if experiment != 'EXP-20261014-reaction-attention':
        raise ValueError('No infrastructure retry registered for this experiment')
    record=read_json(project/'research/experiments'/f'{experiment}.retry1.json')
    if record['retry_number']!=1 or output != (project/record['output']).resolve():
        raise ValueError('Retry identity/output differs')
    old=project/record['previous_output']
    lease=read_json(project/'runs/research-control'/f'{experiment}.lease.json')
    if lease['output']!=str(old) or process_identity(lease['controller_pid'])==lease['start_ticks']:
        raise ValueError('Previous controller still alive or lease identity differs')
    if any((old/name).exists() for name in ('prepared','report.json','active-job.json')):
        raise ValueError('Not a preflight-only failure; reconcile work instead of retrying')
    for path in (old/'frozen-experiment.json',project/'research/experiments'/f'{experiment}.json'):
        # JSON serialization differs between saved manifests and Git source.
        if read_json(path)!=read_json(project/'research/experiments'/f'{experiment}.json'):
            raise ValueError('Retry changes the scientific plan')
    source=project/'research/experiments'/f'{experiment}.json'
    if hashlib.sha256(source.read_text().encode()).hexdigest()!=record['scientific_plan_canonical_sha256']:
        raise ValueError('Retry plan is not pinned')
    return record

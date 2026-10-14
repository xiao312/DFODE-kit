"""Recover the observed pre-optimizer input-schema defect; preserve failed output."""
import argparse
import os
from pathlib import Path
from benchmarks.research_contract.adapter import read_json, arrays, checked_file, digest, verify_evaluator_lock
from benchmarks.research_contract.admission import acquire_lease,process_identity
from benchmarks.research_contract.replay import inputs
from benchmarks.research_contract.sandbox import probe
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from ..hidden_features.campaign import processes_on_gpu
from .prepare import stage,execute


def check_failed_worker(directory):
    if {p.name for p in directory.iterdir()}!={'worker.log','graph.npz'}:
        raise ValueError('Failure contains possible fitting work; do not relaunch')
    if "KeyError: 'features'" not in (directory/'worker.log').read_text():
        raise ValueError('Not the registered schema failure')


def check_telemetry_failure(directory):
    if {p.name for p in directory.iterdir()}!={'worker.log','graph.npz','scaler.npz','feature-audit.npz','pool-state'}:
        raise ValueError('Not the expected first-arm telemetry failure')
    arm=directory/'pool-state'
    if {p.name for p in arm.iterdir()}!={'config.json','gradients.jsonl','progress.jsonl'}:
        raise ValueError('Possible checkpoint/completed work; reconcile instead')
    import json
    progress=[json.loads(line) for line in (arm/'progress.jsonl').read_text().splitlines()]
    if len(progress)!=1 or progress[0]['update']!=1000:
        raise ValueError('Interrupted update count differs')
    gradients=(arm/'gradients.jsonl').read_text().splitlines()
    if len(gradients)!=1000 or json.loads(gradients[-1])['update']!=1000:
        raise ValueError('Partial gradient history differs')
    log=(directory/'worker.log').read_text()
    if 'occupancy_sample' not in log or 'StopIteration' not in log:
        raise ValueError('Not the registered telemetry failure')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--telemetry-repair',action='store_true')
    args=parser.parse_args();project=args.project.resolve();root=args.run.resolve()
    expected=project/'runs/offline-accuracy/reaction-attention-20261014-retry1'
    if root!=expected:raise ValueError('Only the recorded run may be recovered')
    verify_evaluator_lock(project);inputs(project,read_json(project/'research/bindings.json'))
    if source_revision()['dirty']:raise ValueError('Commit recovery source first')
    if not read_json(project/'research/campaign.json')['execution_authorized']:raise ValueError('Execution unauthorized')
    plan=read_json(root/'frozen-experiment.json')
    if plan!=read_json(project/'research/experiments/EXP-20261014-reaction-attention.json'):raise ValueError('Scientific plan changed')
    lease=read_json(project/'runs/research-control/EXP-20261014-reaction-attention-retry1.lease.json')
    if process_identity(lease['controller_pid'])==lease['start_ticks']:raise ValueError('Controller still alive')
    if args.telemetry_repair:
        previous=read_json(root/'schema-recovery-lease.json')
        if process_identity(previous['controller_pid'])==previous['start_ticks']:raise ValueError('Recovery still alive')
    state=read_json(root/'active-job.json')
    if state.get('pid') and process_identity(state['pid'])==state.get('start_ticks'):raise ValueError('Worker still alive')
    if state['status']!='FAILED' or (root/'report.json').exists():raise ValueError('Not the recorded failed run')
    failed=root/str(plan['seeds'][0])
    (check_telemetry_failure if args.telemetry_repair else check_failed_worker)(failed)
    if (root/str(plan['seeds'][1])).exists():raise ValueError('Second seed already has work')
    if 'features' in arrays(root/'prepared'/str(plan['seeds'][0])/'fit-input/training.npz'):
        raise ValueError('Not the diagnosed minimal training schema')
    if processes_on_gpu(args.gpu):print('WAITING_RESOURCE');return
    print('PRECHECKED_INPUT_SCHEMA_RECOVERY',flush=True)
    if not args.execute:return
    failure='telemetry' if args.telemetry_repair else 'schema'
    acquire_lease(root/f'{failure}-recovery-lease.json',dict(start_ticks=process_identity(os.getpid()),source=source_revision()))
    save(root/f'{failure}-failure.json',dict(previous_state=state,source=source_revision(),optimizer_steps_before_failure=1000 if args.telemetry_repair else 0,
         row_presentations_before_failure=200000 if args.telemetry_repair else 0,
         worker_log_sha256=digest(failed/'worker.log'),reason='UUID occupancy compatibility' if args.telemetry_repair else 'Optional cached features absent'))
    # Exact child of the verified run root; archive, never delete failed artifacts.
    failed.rename(root/('failed-20261011-occupancy' if args.telemetry_repair else 'failed-20261011-input-schema'))
    import json
    save(root/f'{failure}-repaired-isolation.json',json.loads(probe(project,root/'probe-input',root/'probe-output',args.gpu)))
    elapsed=0.;records=[]
    parent=project/plan['parent_run'];review=read_json(checked_file(parent/'review.json',plan['parent_review_sha256']))
    report=read_json(checked_file(parent/'report.json',review['report_sha256']))
    for seed in plan['seeds']:
        for name,sha in report['artifacts'].items():
            if name.startswith(f'prepared/{seed}/'):checked_file(parent/name,sha)
        dest=root/'prepared'/str(seed)
        if not dest.exists():dest.mkdir(parents=True);stage(project,dest,seed,plan)
        elapsed+=execute(project,dest,root/str(seed),seed,root/'active-job.json',args.gpu)
        records.append(read_json(root/str(seed)/'result.json'));verify_evaluator_lock(project)
        print(f'SEED_COMPLETE {seed}',flush=True)
    save(root/'active-job.json',dict(status='COMPLETE_AWAITING_RECOUNT'))
    save(root/'report.json',dict(records=records,worker_wall_seconds=elapsed,source=source_revision(),
         artifacts={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
    save(root/'verification.json',dict(report_sha256=digest(root/'report.json')))


if __name__=='__main__':main()

"""Trusted physical scoring, independent thermodynamics and returned-increment recount."""
import argparse
from pathlib import Path
import cantera as ct
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,score,population_rows
from benchmarks.research_contract.gates import comparison,decision
from ..hidden_features.prepare import save
from ..kinetic_features.review import require,recount
from ..paired_base.metrics import align_indices


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();project,root=args.project,args.run
    if (root/'review.json').exists():raise ValueError('Review exists: preserve it')
    verify_evaluator_lock(project);binding=read_json(project/'research/bindings.json');inputs(project,binding)
    report=read_json(checked_file(root/'report.json',read_json(root/'verification.json')['report_sha256']))
    for name,sha in report['artifacts'].items():checked_file(root/name,sha)
    plan=read_json(root/'frozen-experiment.json');policy=read_json(project/'research/campaign.json')['screening']
    prepared,transfer=project/binding['prepared_root'],project/binding['transfer_root']
    active=np.array([s!='AR' for s in binding['species']]);summary=[];scores=[]
    gates={a:{} for a in plan['arms']};timings={a:[] for a in plan['arms']}
    selections=arrays(transfer/'plan/selections.npz');excluded=[];allids={}
    for seed in plan['seeds']:
        with np.load(prepared/str(seed)/'training.npz') as saved:ids=saved['source_indices']
        allids[seed]=ids;excluded.extend(ids[np.r_[np.arange(10000),selections[f'unseen_{seed}']]])
    common=np.setdiff1d(np.intersect1d(*allids.values()),np.unique(excluded))
    ids=np.random.default_rng(plan['selection_seed']).permutation(common)[:plan['fit_rows']]
    for seed in plan['seeds']:
        staged=root/'prepared'/str(seed);trained=root/str(seed);train=arrays(staged/'training-reference.npz')
        np.testing.assert_array_equal(train['source_indices'],ids)
        full=arrays(prepared/str(seed)/'training.npz');ix=align_indices(full['source_indices'],ids)
        for key,value in train.items():np.testing.assert_array_equal(value,full[key][ix])
        expected_features=full['features'][ix]
        del full
        for key,value in arrays(staged/'fit-input/training.npz').items():np.testing.assert_array_equal(value,train[key])
        for name in ('base-weights.pt','preprocessing.npz'):checked_file(staged/'inference-model'/name,digest(prepared/str(seed)/name))
        gas=ct.Solution(str(checked_file(staged/'model/mechanism.yaml',binding['mechanism_sha256'])))
        require(gas.species_names==binding['species'],'Mechanism axis differs')
        checked_file(staged/'inference-model/mechanism.yaml',binding['mechanism_sha256'])
        graph=arrays(trained/'graph.npz');audit=arrays(trained/'feature-audit.npz');scaler=arrays(trained/'scaler.npz')
        np.testing.assert_array_equal(audit['ordinary'],expected_features)
        np.testing.assert_array_equal(graph['product'],gas.product_stoich_coeffs)
        np.testing.assert_array_equal(graph['reactant'],gas.reactant_stoich_coeffs)
        # Reconstruct collider and graph weights separately from metadata().
        collider=np.zeros_like(graph['reactant'])
        for j,reaction in enumerate(gas.reactions()):
            if reaction.third_body is not None:
                body=reaction.third_body
                collider[:,j]=[body.efficiencies.get(s,body.default_efficiency) for s in gas.species_names]
        np.testing.assert_array_equal(graph['collider'],collider)
        total=collider.sum(0);normalized=collider.copy()
        normalized[:,total>0]/=total[total>0]
        support=gas.reactant_stoich_coeffs+gas.product_stoich_coeffs+normalized
        norm=support.sum(1);support[norm>0]/=norm[norm>0,None]
        np.testing.assert_array_equal(graph['pooling'],support)
        np.testing.assert_array_equal(graph['active'],active)
        np.testing.assert_array_equal(audit['source_indices'],ids)
        for j in np.linspace(0,len(ids)-1,32,dtype=int):
            x=train['states'][j];gas.set_unnormalized_mass_fractions(x[2:]);gas.TP=x[0],x[1]
            expected=-(graph['product']-graph['reactant']).T@gas.chemical_potentials/(ct.gas_constant*x[0])
            np.testing.assert_allclose(audit['raw_affinity'][j],expected,rtol=1e-11,atol=1e-10)
        transformed=np.arcsinh(audit['raw_affinity']);sd=transformed.std(0)
        np.testing.assert_array_equal(scaler['mean'],transformed.mean(0))
        np.testing.assert_array_equal(scaler['scale'],np.where(sd>1e-6,sd,1.))
        for name in ('graph.npz','scaler.npz'):checked_file(staged/'inference-model'/name,digest(trained/name))
        records=read_json(trained/'result.json')['records']
        for r in records:
            require(r['updates_completed']==plan['batches'] and r['row_presentations']==plan['batches']*plan['batch_size'] and r['exact_reload'],'Work/replay mismatch')
        for mode in ('pool','attention'):
            pair=[r['initial_weights_sha256'] for r in records if r['arm'].startswith(mode+'-')]
            if pair:require(len(pair)==2 and len(set(pair))==1,'Affinity initialization differs')
        if plan.get('control_run'):
            control=project/plan['control_run']
            oldreview=read_json(checked_file(control/'review.json',plan['control_review_sha256']))
            oldreport=read_json(checked_file(control/'report.json',oldreview['report_sha256']))
            oldresult=read_json(checked_file(control/str(seed)/'result.json',oldreport['artifacts'][f'{seed}/result.json']))
            for name in ('graph.npz','scaler.npz','feature-audit.npz'):
                checked_file(trained/name,oldreport['artifacts'][f'{seed}/{name}'])
            for record in records:
                previous=next(r for r in oldresult['records'] if r['arm']==record['arm'])
                require(record['initial_weights_sha256']==previous['initial_weights_sha256'],'Work-control initialization differs')
                oldcfg=read_json(checked_file(control/str(seed)/record['arm']/'config.json',oldreport['artifacts'][f'{seed}/{record["arm"]}/config.json']))
                cfg=read_json(trained/record['arm']/'config.json')
                for key in set(cfg)|set(oldcfg):
                    if key not in ('experiment','updates','diagnostics_every_updates'):
                        require(cfg.get(key)==oldcfg.get(key),f'Work control changed {key}')
        populations={'training':train};populations.update({p:r for p,r in population_rows(prepared,transfer,str(seed)) if p!='training'})
        for population,rows in populations.items():
            stateonly=arrays(staged/('input-'+population)/'states.npz');require(set(stateonly)=={'states','source_indices'},'Inference boundary')
            for key,value in stateonly.items():np.testing.assert_array_equal(value,rows[key])
            saved=arrays(trained/population/'predictions.npz');np.testing.assert_array_equal(saved['source_indices'],rows['source_indices']);np.testing.assert_array_equal(saved['base'],rows['base'])
            baseline=score(dict(prediction=rows['base'],source_indices=rows['source_indices']),rows,binding['species'],f'{seed}:{population}')
            measured=read_json(trained/population/'result.json');require(measured['label_free_input'],'Missing input check')
            checked_file(trained/population/'predictions.npz',measured['prediction_sha256'])
            for arm in plan['arms']:
                for name in ('weights.pt','config.json'):checked_file(staged/'inference-model'/arm/name,digest(trained/arm/name))
                pred=saved[arm];require(np.isfinite(pred).all(),'Nonfinite prediction')
                np.testing.assert_array_equal(pred[:,~active],rows['base'][:,~active])
                if population=='training':
                    replay=arrays(trained/arm/'training-replay.npz')
                    np.testing.assert_array_equal(replay['prediction'],pred);np.testing.assert_array_equal(replay['coordinate'],saved[arm+'_coordinate'])
                # Independent inverse in long double; no candidate decoder call.
                b=rows['base'][:,active].astype(np.longdouble);c=saved[arm+'_coordinate'].astype(np.longdouble)
                physical=np.longdouble('1e-14')*np.sinh(np.arcsinh(b/np.longdouble('1e-14'))+c)
                tau=1e-15+.1*abs(rows['delta'][:,active])
                require(np.max(abs(physical-pred[:,active])/tau)<1e-5,'Decoded increments differ')
                result=score(dict(prediction=pred,source_indices=rows['source_indices']),rows,binding['species'],f'{seed}:{population}')
                recount(pred,rows['delta'],rows['base'],result,active,rows['states'][:,2:]);g=result['groups']['all']
                t=next(r for r in measured['records'] if r['arm']==arm)
                ratio=float(np.median(t['timings']['candidate'])/np.median(t['timings']['anchor']));require(ratio==t['ratio'],'Timing differs')
                summary.append(dict(seed=seed,population=population,arm=arm,acceptance=g['acceptance'],base=g['base_acceptance'],repairs=g['repaired'],damage=g['damaged'],damage_rate=g['damage_rate_among_base_successes'],p99=g['normalized_error']['p99'],negative=result['endpoint_diagnostics']['negative_components'],latency_ratio=ratio))
                scores.append(dict(seed=seed,population=population,arm=arm,candidate=result,baseline=baseline))
                if population!='training':
                    for comparator in ('anchor','incumbent'):gates[arm][f'{seed}:{population}:{comparator}']=comparison(result,baseline,policy)
                    timings[arm].append(ratio)
    keys={f'{s}:{p}:{c}' for s in plan['seeds'] for p in ('same_source_unseen','development') for c in ('anchor','incumbent')}
    verdict={a:decision(gates[a],keys,max(timings[a]),policy) for a in plan['arms']}
    interrupted=[]
    for kind in ('schema','telemetry'):
        path=root/f'{kind}-failure.json'
        if path.exists():
            failed=read_json(path)
            interrupted.append(dict(kind=kind,record_sha256=digest(path),
                worker_wall_seconds=failed['previous_state']['elapsed_seconds'],
                optimizer_steps=failed['optimizer_steps_before_failure'],
                row_presentations=(0 if failed['optimizer_steps_before_failure']==0
                                   else failed['row_presentations_before_failure'])))
    resources=dict(successful_worker_wall_seconds=report['worker_wall_seconds'],
        interrupted_attempts=interrupted,
        total_observed_worker_wall_seconds=report['worker_wall_seconds']+sum(r['worker_wall_seconds'] for r in interrupted),
        interrupted_row_presentations=sum(r['row_presentations'] for r in interrupted),
        gpu_allocation_seconds=None,
        note='Worker wall time is not GPU allocation time. Preflight time was not instrumented. Outside-controller contention checks are authoritative; PID-namespace occupancy counts can include the worker itself.')
    save(root/'protected-scores.json',dict(records=scores));verify_evaluator_lock(project)
    save(root/'review.json',dict(status='verified',summary=summary,decision=verdict,incumbent_unchanged=True,confirmation_opened=False,
        report_sha256=digest(root/'report.json'),scores_sha256=digest(root/'protected-scores.json'),resources=resources))
    print(summary);print(verdict)


if __name__=='__main__':main()

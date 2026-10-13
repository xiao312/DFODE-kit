"""Protected physical scores plus independent reconstruction and provenance."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.special import erf
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,score,population_rows
from benchmarks.research_contract.gates import comparison,decision
from ..hidden_features.prepare import save
from ..kinetic_features.review import require,recount
from ..paired_base.metrics import align_indices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();project,root=args.project,args.run
    if (root/'review.json').exists():raise ValueError('Review exists')
    verify_evaluator_lock(project);binding=read_json(project/'research/bindings.json');inputs(project,binding)
    report=read_json(checked_file(root/'report.json',read_json(root/'verification.json')['report_sha256']))
    for name,sha in report['artifacts'].items():checked_file(root/name,sha)
    plan=read_json(root/'frozen-experiment.json');policy=read_json(project/'research/campaign.json')['screening']
    prepared,transfer=project/binding['prepared_root'],project/binding['transfer_root'];active=np.array([s!='AR' for s in binding['species']])
    summary=[];scores=[];checks=[];gates={a:{} for a in plan['arms']};timing={a:[] for a in plan['arms']}
    selections=arrays(transfer/'plan/selections.npz');excluded=[];allids={}
    for seed in plan['seeds']:
        with np.load(prepared/str(seed)/'training.npz') as z:ids=z['source_indices']
        allids[seed]=ids;excluded.extend(ids[np.r_[np.arange(10000),selections[f'unseen_{seed}']]])
    common=np.setdiff1d(np.intersect1d(*allids.values()),np.unique(excluded))
    ids=np.random.default_rng(plan['selection_seed']).permutation(common)[:plan['fit_rows']]
    for seed in plan['seeds']:
        staged=root/'prepared'/str(seed);trained=root/str(seed);train=arrays(staged/'training-reference.npz')
        np.testing.assert_array_equal(train['source_indices'],ids)
        full=arrays(prepared/str(seed)/'training.npz');ix=align_indices(full['source_indices'],ids)
        for key,value in train.items():np.testing.assert_array_equal(value,full[key][ix])
        del full
        for name in ('base-weights.pt','preprocessing.npz'):checked_file(staged/'inference-model'/name,digest(prepared/str(seed)/name))
        scales=arrays(trained/'scales.npz');checked_file(staged/'inference-model/scales.npz',digest(trained/'scales.npz'))
        record=read_json(trained/'result.json');require(record['initial_interval_pass_parity'],'Missing initial interval check')
        initial_hashes=[]
        for r in record['records']:
            require(r['updates_completed']==plan['batches'] and r['row_presentations']==plan['batches']*plan['batch_size'],'Work differs')
            require(r['parameter_count']==plan['parameters'] and r['exact_reload'],'Head/replay differs')
            initial_hashes.append(r['initial_weights_sha256'])
        require(len(set(initial_hashes))==1,'Initialization differs')
        target=arrays(trained/'targets.npz');np.testing.assert_array_equal(target['source_indices'],ids)
        # Reconstruct finite physical bounds and normalized interval independently
        # with long-double power/log arithmetic, avoiding the target helper.
        prep=arrays(prepared/str(seed)/'preprocessing.npz');y=train['states'][:,2:][:,active];d=train['delta'][:,active]
        tau=1e-15+.1*abs(d);physical=[np.maximum(-y,d-tau),d+tau,d]
        encoded=[]
        for value in physical:
            yy=y.astype(np.longdouble);vv=value.astype(np.longdouble);end=yy+vv
            change=np.empty_like(yy);regular=(yy>0)&(end>0)
            small=regular&(abs(vv)<.5*yy)
            change[small]=yy[small]**np.longdouble('.1')*np.expm1(np.longdouble('.1')*np.log1p(vv[small]/yy[small]))/np.longdouble('.1')
            rest=~small;change[rest]=(np.maximum(end[rest],0)**np.longdouble('.1')-yy[rest]**np.longdouble('.1'))/np.longdouble('.1')
            coordinate=2*np.sign(change)*np.sqrt(abs(change)/np.longdouble('1e-6'))
            encoded.append((coordinate.astype(float)-prep['base_y_offset'])/prep['base_y_scale'])
        lo,hi,point=encoded;floor=128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(lo),abs(hi)));width=np.maximum((hi-lo)/2,floor)
        np.testing.assert_allclose(scales['output_scale'],np.median(width,axis=0),rtol=2e-9,atol=1e-13)
        alpha=scales['output_scale'];packed=target['target'];raw=target['raw']
        finite=d-tau>-y
        np.testing.assert_array_equal(np.isneginf(packed[...,0]),~finite)
        # Compare in fractions of component width, not raw huge coordinates.
        discrepancies=[np.max(abs((packed[...,0]*alpha+raw-lo)[finite])/width[finite]),
                       np.max(abs(packed[...,1]*alpha+raw-hi)/width),np.max(abs(packed[...,3]*alpha+raw-point)/width)]
        require(max(discrepancies)<1e-5,'Independent target interval differs')
        audit=arrays(trained/'hidden-audit.npz');ai=audit['indices'];state=train['states'][ai]
        weights={k:v.double().numpy() for k,v in torch.load(prepared/str(seed)/'base-weights.pt',weights_only=True,map_location='cpu').items()}
        h=((np.column_stack([state[:,:2],state[:,2:]**.1])-prep['base_x_offset'])/prep['base_x_scale']).astype(np.float32).astype(float)
        for layer in (0,2,4,6):
            v=h@weights[f'{layer}.weight'].T+weights[f'{layer}.bias'];h=.5*v*(1+erf(v/np.sqrt(2)))
        np.testing.assert_allclose(h,audit['hidden'],rtol=2e-4,atol=2e-4)
        normalized=((audit['hidden'].astype(float)-scales['hidden_offset'])/scales['hidden_scale']).astype(np.float32)
        np.testing.assert_array_equal(normalized,audit['features'])
        checks.append(dict(seed=seed,independent_target_max_width_error=max(discrepancies),prechecks=record['prechecks']))
        populations={'training':train}
        populations.update({p:r for p,r in population_rows(prepared,transfer,str(seed)) if p!='training'})
        for population,rows in populations.items():
            stateonly=arrays(staged/('input-'+population)/'states.npz');require(set(stateonly)=={'states','source_indices'},'Inference boundary')
            for key,value in stateonly.items():np.testing.assert_array_equal(value,rows[key])
            saved=arrays(trained/population/'predictions.npz');np.testing.assert_array_equal(saved['source_indices'],rows['source_indices']);np.testing.assert_array_equal(saved['base'],rows['base'])
            measured=read_json(trained/population/'result.json');require(measured['exact_reload_replay'],'Reload missing')
            baseline=score(dict(prediction=rows['base'],source_indices=rows['source_indices']),rows,binding['species'],f'{seed}:{population}')
            for arm in plan['arms']:
                for name in ('weights.pt','config.json'):checked_file(staged/'inference-model'/arm/name,digest(trained/arm/name))
                pred=saved[arm];require(np.isfinite(pred).all(),'Nonfinite prediction')
                np.testing.assert_array_equal(pred[:,~active],rows['base'][:,~active])
                if population=='training':
                    replay=arrays(trained/arm/'training-replay.npz')
                    np.testing.assert_array_equal(replay['prediction'],pred);np.testing.assert_array_equal(replay['coordinate'],saved[arm+'_coordinate'])
                    hw={k:v.double().numpy() for k,v in torch.load(trained/arm/'weights.pt',weights_only=True,map_location='cpu').items()}
                    cpu=normalized.astype(float)@hw['0.weight'].T+hw['0.bias']
                    np.testing.assert_allclose(cpu,replay['coordinate'][ai],rtol=3e-5,atol=3e-6)
                    corrected=raw+alpha*replay['coordinate']
                    z=corrected*prep['base_y_scale']+prep['base_y_offset']
                    change=1e-6*.25*z*abs(z);initial_power=y**.1
                    power=initial_power+.1*change;domain=power<0
                    independent=np.empty_like(y);zero=y==0;positive=(y>0)&(power>0)
                    independent[zero]=np.maximum(power[zero],0)**10
                    independent[(y>0)&~positive]=-y[(y>0)&~positive]
                    ratio=.1*change[positive]/initial_power[positive]
                    logarithm=np.empty_like(ratio);small=abs(ratio)<.5
                    logarithm[small]=np.log1p(ratio[small])
                    logarithm[~small]=np.log(power[positive][~small])-np.log(initial_power[positive][~small])
                    independent[positive]=y[positive]*np.expm1(10*logarithm)
                    np.testing.assert_array_equal(domain,saved[arm+'_domain_correction'])
                    require(np.max(abs(independent-pred[:,active])/tau)<1e-6,'Independent reconstruction differs')
                    accepted=(replay['coordinate']>=packed[...,0])&(replay['coordinate']<=packed[...,1])
                    actual=abs(pred[:,active]-d)<=tau
                    require(np.array_equal(accepted,actual),'Final interval membership differs from physical pass')
                result=score(dict(prediction=pred,source_indices=rows['source_indices']),rows,binding['species'],f'{seed}:{population}')
                recount(pred,rows['delta'],rows['base'],result,active,rows['states'][:,2:]);g=result['groups']['all']
                time_row=next(r for r in measured['records'] if r['arm']==arm)
                ratio=float(np.median(time_row['timings']['candidate'])/np.median(time_row['timings']['anchor']))
                require(ratio==time_row['ratio'],'Timing recount differs')
                summary.append(dict(seed=seed,population=population,arm=arm,acceptance=g['acceptance'],base=g['base_acceptance'],repairs=g['repaired'],damage=g['damaged'],damage_rate=g['damage_rate_among_base_successes'],p99=g['normalized_error']['p99'],negative=result['endpoint_diagnostics']['negative_components'],latency_ratio=ratio))
                scores.append(dict(seed=seed,population=population,arm=arm,candidate=result,baseline=baseline))
                if population!='training':
                    for comparator in ('anchor','incumbent'):gates[arm][f'{seed}:{population}:{comparator}']=comparison(result,baseline,policy)
                    timing[arm].append(ratio)
    keys={f'{s}:{p}:{c}' for s in plan['seeds'] for p in ('same_source_unseen','development') for c in ('anchor','incumbent')}
    verdict={a:decision(gates[a],keys,max(timing[a]),policy) for a in plan['arms']}
    save(root/'protected-scores.json',dict(records=scores));verify_evaluator_lock(project)
    save(root/'review.json',dict(status='verified',summary=summary,decision=verdict,checks=checks,incumbent_unchanged=True,
         confirmation_opened=False,report_sha256=digest(root/'report.json'),scores_sha256=digest(root/'protected-scores.json')))
    print(summary);print(verdict)


if __name__=='__main__':main()

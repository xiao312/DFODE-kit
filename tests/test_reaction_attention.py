"""Small synthetic checks; neural tests use CPU verification, never CPU fits."""
import json
from pathlib import Path
import numpy as np
import pytest
ct=pytest.importorskip('cantera')
from benchmarks.offline_accuracy.paper_baseline.reaction_attention.chemistry import metadata, affinity, fit_scaler, pack, normalize_columns


def problem():
    gas=ct.Solution('h2o2.yaml');gas.TPX=1300,2*ct.one_atm,'H2:2,O2:1,N2:3.76'
    return gas,np.r_[gas.T,gas.P,gas.Y][None,:]


def test_stoichiometry_and_colliders():
    gas,_=problem();g=metadata(gas,gas.species_names)
    np.testing.assert_array_equal(g['elements']@(g['product']-g['reactant']),0)
    np.testing.assert_allclose(g['pooling'][g['active']].sum(1),1)
    for j,r in enumerate(gas.reactions()):
        if r.third_body is not None:
            for i,name in enumerate(gas.species_names):
                assert g['collider'][i,j]==r.third_body.efficiencies.get(name,r.third_body.default_efficiency)
        else:assert not g['collider'][:,j].any()
    with pytest.raises(ValueError):metadata(gas,gas.species_names[::-1])


def test_affinity_units_identity_zero_species_and_replay():
    gas,states=problem();a=affinity(gas,states)
    expected=-((gas.product_stoich_coeffs-gas.reactant_stoich_coeffs).T@gas.chemical_potentials)/(ct.gas_constant*gas.T)
    np.testing.assert_allclose(a[0],expected,rtol=1e-12,atol=1e-11)
    assert np.isfinite(a).all()
    gas.basis='molar';np.testing.assert_array_equal(affinity(gas,states),a)
    for broken in (states*float('nan'),states.copy()):
        if np.isfinite(broken).all():broken[:,2:]*=1.1
        with pytest.raises(ValueError):affinity(gas,broken)


def test_scaler_and_axes():
    raw=np.array([[0.,2.],[3.,2.]])
    scaler=fit_scaler(raw);np.testing.assert_array_equal(scaler['scale'][1:],1)
    out=pack(np.zeros((2,3)),raw,scaler)
    np.testing.assert_allclose(out[:,-2:].mean(0),0,atol=1e-7)
    assert out.dtype==np.float32
    with pytest.raises(ValueError):pack(np.zeros((1,3)),raw,scaler)
    with pytest.raises(ValueError):normalize_columns(np.array([[-1.]]))
    with pytest.raises(ValueError):fit_scaler(np.array([[np.nan]]))
    np.testing.assert_array_equal(normalize_columns(np.zeros((2,3))),0)


@pytest.mark.parametrize('mode',['pool','attention'])
def test_zero_head_paired_init_gradients_and_replay(mode):
    torch=pytest.importorskip('torch')
    from benchmarks.offline_accuracy.paper_baseline.reaction_attention.model import ReactionHead
    gas,_=problem();graph=metadata(gas,gas.species_names)
    cfg=dict(seed=20261011,width=32,heads=2,arm=mode+'-state')
    a=ReactionHead(cfg,graph);b=ReactionHead(dict(cfg,arm=mode+'-affinity'),graph)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    x=torch.randn(3,2+gas.n_species+int(graph['active'].sum())+gas.n_reactions)
    torch.testing.assert_close(a(x),torch.zeros(3,int(graph['active'].sum())),rtol=0,atol=0)
    with torch.no_grad():a.output.weight.fill_(.02);b.output.weight.fill_(.02)
    altered=x.clone();altered[:,-gas.n_reactions:]+=1
    torch.testing.assert_close(a(x),a(altered),rtol=0,atol=0)
    # Nonuniform readout avoids the LayerNorm constant-sum identity.
    with torch.no_grad():a.output.weight.normal_();b.output.weight.copy_(a.output.weight)
    assert not torch.allclose(b(x),b(altered))
    a(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
    clone=ReactionHead(cfg,graph);clone.load_state_dict(a.state_dict())
    torch.testing.assert_close(clone(x),a(x),rtol=0,atol=0)
    if mode=='attention':
        q,r=a.tokens(x);w=a.attention_weights(q,r)
        torch.testing.assert_close(w.sum(-1),torch.ones_like(w.sum(-1)))
        assert torch.count_nonzero(w[...,a.pooling==0])==0


def test_registered_factorial_and_work():
    plan=json.loads(Path('research/experiments/EXP-20261014-reaction-attention.json').read_text())
    assert set(plan['arms'])=={f'{m}-{a}' for m in ('pool','attention') for a in ('state','affinity')}
    assert plan['fit_rows']%plan['batch_size']==0
    assert plan['batches']*plan['batch_size']==600000
    assert not plan['confirmation_access']

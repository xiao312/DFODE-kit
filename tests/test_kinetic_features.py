import json
from pathlib import Path
import cantera as ct
import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.kinetic_features.model import rate_features,fit_scaler,augment
from benchmarks.offline_accuracy.improve.physics_prior import frozen_step


def test_scaler_training_only_and_constant():
    fitting=np.array([[1.,2.],[3.,2.]])
    scaler=fit_scaler(fitting)
    np.testing.assert_array_equal(scaler["mean"],[2.,2.])
    np.testing.assert_array_equal(scaler["scale"],[1.,1e-6])
    x=augment(np.ones((1,3),np.float32),np.array([[5.,2.]]),scaler)
    np.testing.assert_array_equal(x,[[1,1,1,3,0]])
    assert x.dtype==np.float32


def test_bad_features_fail_closed():
    with pytest.raises(ValueError):fit_scaler(np.array([[np.nan]]))
    with pytest.raises(ValueError):augment(np.ones((2,3)),np.ones((1,2)),dict(mean=np.ones(2),scale=np.ones(2)))


def test_rate_units_species_axis_and_replay():
    gas=ct.Solution("h2o2.yaml"); gas.TPX=1300,2*ct.one_atm,"H2:2,O2:1,N2:3.76"
    state=np.r_[gas.T,gas.P,gas.Y][None,:]; active=np.array([s!="AR" for s in gas.species_names])
    expected=1e-6*gas.net_production_rates*gas.molecular_weights/gas.density_mass
    coordinate,raw=rate_features(gas,state,active)
    np.testing.assert_allclose(raw[0,:active.sum()],expected[active],rtol=2e-14,atol=1e-35)
    np.testing.assert_array_equal(coordinate,np.arcsinh(raw/1e-14))
    np.testing.assert_array_equal(rate_features(gas,state,active)[1],raw)
    gas.basis="molar"
    np.testing.assert_array_equal(rate_features(gas,state,active)[1],raw)


def test_invalid_state_not_normalized():
    gas=ct.Solution("h2o2.yaml"); gas.TPX=1200,ct.one_atm,"H2:2,O2:1"
    state=np.r_[gas.T,gas.P,gas.Y][None,:]; state[0,2:]*=1.1
    with pytest.raises(ValueError):rate_features(gas,state,np.ones(gas.n_species,bool))


def test_analytic_decay_and_zero_loss():
    y=np.array([.4,.6]); rate=np.array([-.8,.8]); loss=np.array([.8,0.])
    result=frozen_step(y,rate,loss,.2,"frozen-exponential")
    np.testing.assert_allclose(result,[.4*np.expm1(-.4),.16],rtol=1e-14)


def test_registered_parameter_counts_and_roles():
    plan=json.loads(Path("research/experiments/EXP-20261013-kinetic-features.json").read_text())
    for arm,widths in plan["widths"].items():
        sizes=[235 if arm=="kinetic" else 119,*widths,58]
        assert sum((a+1)*b for a,b in zip(sizes[:-1],sizes[1:]))==plan["parameters"][arm]
    assert plan["fit_rows"]+plan["calibration_rows"]+plan["check_rows"]==50000
    assert plan["batches"]*plan["batch_size"]==60000000

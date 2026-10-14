import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.resolved_fit.data import selected_rows,scores
from benchmarks.offline_accuracy.paper_baseline.resolved_fit.plan import configuration,SEEDS,WIDTHS,OBJECTIVES


def fixture_rows():
    training = dict(states=np.array([[900.,1e5,.1,.9],[1000.,1e5,.2,.8]]),
                    source_indices=np.array([7,9]),delta=np.zeros((2,2)))
    records = [dict(split="train",source_row=i,status="checked",reference_delta=[1e-8,1e-8],
                    uncertainty_estimate=[u,0.]) for i,u in ((9,0.),(7,1e-5))]
    return training,dict(records=records)


def test_selection_excludes_nontraining_and_rejects_missing_or_duplicate():
    rows,audit = fixture_rows()
    # Make one resolved component per row, with the first species unresolved on row 7.
    sample = selected_rows(rows,audit,np.array([True,True]),expected_rows=2)
    assert sample["source_indices"].tolist()==[7,9]
    assert not sample["resolved"][0,0] and sample["resolved"][1,0]
    audit["records"].append(dict(audit["records"][0],split="validation",source_row=100))
    np.testing.assert_array_equal(selected_rows(rows,audit,[True,True],2)["states"],sample["states"])
    with pytest.raises(ValueError,match="no resolved"):
        selected_rows(rows,audit,[True,False],2)
    audit["records"][1]["source_row"] = 99
    with pytest.raises(ValueError,match="absent"):
        selected_rows(rows,audit,[True,True],2)
    audit["records"][1]["source_row"] = 9
    with pytest.raises(ValueError,match="distinct"):
        selected_rows(rows,audit,[True,True],2)


def test_scores_do_not_call_partial_rows_whole_states():
    reference = np.array([[1e-12,1e-6,0.],[1e-12,1e-6,0.]])
    prediction = reference.copy()
    prediction[0,1] = 100.
    mask = np.array([[True,False,False],[True,True,False]])
    result = scores(prediction,reference,mask,np.zeros_like(reference),["H2","O2","AR"])
    assert result["resolved"]["acceptance"]==1
    assert result["nontrivial"]["components"]==3
    assert result["resolved_subset_rows_passed"]==2
    assert result["fully_resolved_rows"]==result["whole_state_pass_count"]==1


def test_fixed_matrix():
    configs = [configuration(s,w,o) for s in SEEDS for w in WIDTHS for o in OBJECTIVES]
    assert len(configs)==8
    assert all(c["updates"]==6000 and c["batch_size"]==16 and c["decoder"]=="asinh" for c in configs)
    with pytest.raises(ValueError):
        configuration(SEEDS[0],1024,"direct")


def test_masked_objective_ignores_unresolved_values():
    torch = pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.resolved_fit.fit import objective
    c = torch.tensor([[.2,3.]],dtype=torch.float32,requires_grad=True)
    b = torch.tensor([[1e-12,1e-8]],dtype=torch.float64)
    mask = torch.tensor([[True,False]])
    target = torch.tensor([[.1,100.]],dtype=torch.float64)
    for method in OBJECTIVES:
        config = configuration(SEEDS[0],256,method)
        loss = objective(c,b,b,mask,target,config)
        loss.backward()
        assert c.grad[0,0]!=0 and c.grad[0,1]==0
        c.grad.zero_()
    assert objective(c,b,b,mask,target,configuration(SEEDS[0],256,"direct")).item()==pytest.approx(.01)


@pytest.mark.parametrize("method",OBJECTIVES)
def test_gpu_fit_and_replay(method,tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU training required")
    from benchmarks.offline_accuracy.paper_baseline.resolved_fit.fit import fit
    torch.set_num_threads(1)
    config = dict(configuration(SEEDS[0],256,method),widths=[8]*3,updates=3,diagnostics_every_updates=1)
    base = np.array([[1e-8,0.],[1e-9,0.]])
    sample = dict(reference=np.array([[1e-12,0.],[1e-11,0.]]),resolved=np.array([[True,False],[True,False]]))
    result = fit(np.array([[1.,2.],[3.,4.]],dtype=np.float32),base,sample,
                 np.array([True,False]),["H2","AR"],config,tmp_path/method)
    assert result["exact_head_replay"] and result["cuda_numpy_pass_agreement"]
    assert result["zero_correction_identity"] and result["raw_gradient_norm"]["max"]>0

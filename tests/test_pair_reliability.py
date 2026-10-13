import numpy as np
from benchmarks.offline_accuracy.paper_baseline.paired_base.reliability import groups,fit,predict


def test_groups_exclude_zero_growth_and_sign_reversal():
    b=np.array([[1.,1.,0.,1.,-1.,1.]])
    q=np.array([[2.,-1.,1.,0.,-.5,.01]])
    np.testing.assert_array_equal(groups(b,q),[[-1,-1,-1,-1,2,3]])


def test_training_table_repairs_but_query_does_not_use_truth():
    b=np.ones((200,1));q=b*.5;d=np.ones_like(b)
    d[:100]=q[:100]
    # Switching damages every previously correct component, so refuse.
    table,_=fit(b,q,d);assert not table.any()
    np.testing.assert_array_equal(predict(b,q,table),b)
    # A mixed bin with ample repairs and <=2% damage can be selected.
    q[:100]=.99;b[:100]=.995;d[:100]=.99
    table,_=fit(b,q,d,minimum=50)
    assert not table.any()  # No net repair in well-fitted bin; zero basepass in failing bin.


def test_selector_preserves_sign_and_magnitude():
    b=np.array([[1.,-1.,0.],[2.,-2.,1e-30]])
    q=np.array([[.1,-.1,1.],[3.,2.,1e-31]])
    y=predict(b,q,np.ones((3,4),dtype=bool))
    np.testing.assert_array_equal(np.sign(y),np.sign(b));assert np.all(abs(y)<=abs(b))


def test_table_can_select_supported_net_repair():
    b=np.ones((200,1));q=b*.9;d=np.full_like(b,.95);d[:100]=.82
    table,records=fit(b,q,d)
    assert table[0,1]
    np.testing.assert_array_equal(predict(b,q,table),q)
    chosen=[r for r in records if r["selected"]]
    assert len(chosen)==1 and chosen[0]["repairs"]==100 and chosen[0]["damage"]==0

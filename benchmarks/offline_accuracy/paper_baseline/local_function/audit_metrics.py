"""Independent physical recounts, without importing the protected grader."""
import numpy as np


def gate_failures(d, b, y, state_y, policy, events, species):
    """Reconstruct the fixed gate on complete, finite, aligned active arrays."""
    e=abs(y-d)/(1e-15+.1*abs(d));be=abs(b-d)/(1e-15+.1*abs(d))
    passed=e<=1;old=be<=1;nt=abs(d)>1e-15/.9
    weak=nt&(abs(d)<=1e-12);active=abs(d)>1e-12
    failures=[]
    def check(name,condition):
        if not condition:failures.append(name)
    check("overall_gain",passed.mean()-old.mean()>=policy["minimum_gain_fraction"])
    check("nontrivial_gain",passed[nt].mean()-old[nt].mean()>=0)
    for name,mask in [("weak_nontrivial",weak),("active_increment",active)]:
        if mask.sum()<policy["minimum_group_components_for_pilot_gate"]:
            raise ValueError("Insufficient group support in recount")
        check(name+"_gain",passed[mask].mean()-old[mask].mean()>=-policy["primary_group_max_drop_fraction"])
    for i,name in enumerate(species):
        check("species_gain:"+name,passed[:,i].mean()-old[:,i].mean()>=-policy["supported_species_max_drop_fraction"])
    check("damage",(~passed&old).sum()/old.sum()<=policy["maximum_damage_rate"])
    for name,mask in [("all",np.ones_like(passed)),("weak_nontrivial",weak)]:
        check(name+"_p99",np.quantile(e[mask],.99)<=policy["maximum_p99_ratio"]*np.quantile(be[mask],.99))
    check("wrong_sign",((np.sign(y)!=np.sign(d))&nt).sum()<=((np.sign(b)!=np.sign(d))&nt).sum())
    for event in events:
        key=(event["epsilon"],event["eta"]);mask=abs(d)<=key[0]
        check(f"event:{key}",(mask&(abs(y)>=key[1])).sum()<=(mask&(abs(b)>=key[1])).sum())
    check("negative_endpoints",((state_y+y)<0).sum()<=((state_y+b)<0).sum())
    return sorted(failures)


def damage_slices(d,b,y,snapshots):
    tau=1e-15+.1*abs(d);old_error=abs(b-d)/tau
    old=old_error<=1;passed=abs(y-d)<=tau
    def counts(mask):
        return dict(components=int(mask.sum()),base_passes=int((mask&old).sum()),
                    passes=int((mask&passed).sum()),damaged=int((mask&old&~passed).sum()),
                    repaired=int((mask&~old&passed).sum()))
    margins={}
    for lo,hi in [(-1.,.1),(.1,.5),(.5,.8),(.8,1.)]:
        margins[f"({lo},{hi}]"]=counts((old_error>lo)&(old_error<=hi))
    sources={str(s):counts(np.broadcast_to((snapshots==s)[:,None],d.shape)) for s in np.unique(snapshots)}
    return dict(base_margin=margins,snapshot=sources)

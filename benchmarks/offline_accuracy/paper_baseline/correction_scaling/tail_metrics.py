"""Component-level diagnostics; no fitted choices use development data."""
import numpy as np
from ..decoder_loss.coordinates import decode


def relationships(base,truth):
    labels = np.full(base.shape,"same-sign equal magnitude",dtype="<U32")
    labels[abs(base)>abs(truth)] = "same-sign overprediction"
    labels[abs(base)<abs(truth)] = "same-sign underprediction"
    labels[np.sign(base)*np.sign(truth)<0] = "sign reversal"
    labels[(base==0)|(truth==0)] = "one or both zero"
    return labels


def physical_scores(prediction,truth,base):
    allowance = 1e-15+.1*abs(truth)
    ratio = abs(prediction-truth)/allowance
    passed,old = ratio<=1,abs(base-truth)<=allowance
    nontrivial = abs(truth)>1e-15/.9
    return dict(components=int(truth.size),acceptance=float(passed.mean()),
        nontrivial_acceptance=float(passed[nontrivial].mean()) if nontrivial.any() else None,
        normalized_error_p99=float(np.quantile(ratio,.99)),
        pass_to_fail=int((old&~passed).sum()),fail_to_pass=int((~old&passed).sum()))


def constant_scores(base,truth,target,mean,config):
    coordinate = np.broadcast_to(mean,target.shape)
    prediction = decode(base,coordinate,config)
    return dict(physical_scores(prediction,truth,base),
        coordinate_mse=float(np.mean((coordinate-target)**2)),
        zero_coordinate_mse=float(np.mean(target**2)))


def selected_tail(target):
    count = max(1,target.size//100)
    # Stable tie breaking makes the selected component list reproducible.
    indices = np.argsort(-(target.ravel()**2),kind="stable")[:count]
    return np.unravel_index(indices,target.shape)


def grouped_tail(values,row_indices,squares,total):
    return [dict(group=str(value),components=int((values==value).sum()),
                 distinct_states=len(np.unique(row_indices[values==value])),
                 initial_mse_share=float(squares[values==value].sum()/total) if total else 0.)
            for value in np.unique(values)]

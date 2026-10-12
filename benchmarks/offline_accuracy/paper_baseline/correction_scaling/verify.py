"""Arithmetic agreement in physical tolerance units, including near-zero predictions."""
import numpy as np
from ..decoder_loss.coordinates import decode,torch_decode


def compare_decoders(first,second,reference):
    allowance = 1e-15+.1*abs(reference)
    discrepancy = abs(first-second)/allowance
    if not np.isfinite(discrepancy).all() or discrepancy.max()>1e-9:
        raise ValueError("Decoder discrepancy exceeds numerical-check allowance")
    np.testing.assert_array_equal(abs(first-reference)<=allowance,abs(second-reference)<=allowance)
    return dict(max_discrepancy_in_allowances=float(discrepancy.max()),identical_pass_decisions=True,
                check_limit=1e-9,physical_rule_unchanged=True)


def decoder_agreement(base,coordinate,reference,config):
    import torch
    with torch.no_grad():
        gpu = torch_decode(torch.as_tensor(base,device="cuda"),torch.as_tensor(coordinate,device="cuda"),config).cpu().numpy()
    return compare_decoders(decode(base,coordinate,config),gpu,reference)

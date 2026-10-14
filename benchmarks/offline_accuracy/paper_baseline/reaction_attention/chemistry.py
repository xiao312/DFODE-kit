"""Pinned mechanism metadata and state-only thermodynamics. No integration."""
import numpy as np
from benchmarks.flame_conditioning.chemistry import set_state


def normalize_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Expected finite nonnegative incidence")
    total = matrix.sum(0)
    return np.divide(matrix, total, out=np.zeros_like(matrix), where=total > 0)


def metadata(gas, species):
    if gas.species_names != list(species):
        raise ValueError("Mechanism species ordering differs")
    reactant = gas.reactant_stoich_coeffs.copy()
    product = gas.product_stoich_coeffs.copy()
    collider = np.zeros_like(reactant)
    reversible, types = [], []
    for j, reaction in enumerate(gas.reactions()):
        reversible.append(reaction.reversible)
        types.append(reaction.reaction_type)
        if reaction.third_body is not None:
            body = reaction.third_body
            collider[:, j] = [body.efficiencies.get(s, body.default_efficiency) for s in species]
    elements = np.array([[gas.n_atoms(i, e) for i in range(gas.n_species)] for e in range(gas.n_elements)])
    if not np.allclose(elements @ (product-reactant), 0, atol=1e-12, rtol=0):
        raise ValueError("Mechanism does not conserve elements")
    type_names = sorted(set(types))
    type_flags = np.eye(len(type_names))[np.array([type_names.index(t) for t in types])]
    constants = np.column_stack([reversible, collider.sum(0)>0, reactant.sum(0), product.sum(0), type_flags])
    # A collider influences a reaction even if its net stoichiometric coefficient
    # is zero. Keep its separate input path and aggregation support.
    support = reactant + product + normalize_columns(collider)
    sums = support.sum(1, keepdims=True)
    active = np.array([s != "AR" for s in species])
    if np.any(sums[active] <= 0):
        raise ValueError("Active species without a reaction participant path")
    pooling = np.divide(support, sums, out=np.zeros_like(support), where=sums > 0)
    return dict(reactant=reactant, product=product, collider=collider,
                elements=elements, constants=constants, pooling=pooling, active=active)


def affinity(gas, states):
    """Return A/(RT). Cantera uses finite limiting potentials for absent species."""
    import cantera as ct
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != gas.n_species+2 or not len(states):
        raise ValueError("Expected nonempty ordered T/P/Y states")
    result = []
    gas.basis = "mass"
    for row in states:
        set_state(gas, dict(T=row[0], P=row[1], Y=row[2:]))
        result.append(-gas.delta_gibbs/(ct.gas_constant*gas.T))
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite affinity: no clipping fallback")
    return result


def fit_scaler(raw):
    values = np.arcsinh(np.asarray(raw, dtype=float))
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid affinity calibration population")
    scale = values.std(0)
    return dict(mean=values.mean(0), scale=np.where(scale > 1e-6, scale, 1.))


def pack(ordinary, raw, scaler):
    ordinary, raw = np.asarray(ordinary), np.asarray(raw)
    if raw.ndim != 2 or len(raw) != len(ordinary) or raw.shape[1:] != scaler['mean'].shape:
        raise ValueError("Affinity feature axes differ")
    if not np.isfinite(scaler['scale']).all() or np.any(scaler['scale'] <= 0):
        raise ValueError("Invalid frozen affinity scale")
    out = np.column_stack([ordinary, (np.arcsinh(raw)-scaler['mean'])/scaler['scale']]).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite features")
    return out

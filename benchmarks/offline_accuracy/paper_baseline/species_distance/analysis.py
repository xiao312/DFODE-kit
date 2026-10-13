"""Score fixed species graphs; query labels never select a neighbor."""
import numpy as np
from ..decoder_loss.coordinates import decode
from ..local_predictability.analysis import SCALE, summarize


def transfer_species(library, query_base, indices, active, mode):
    """indices has shape (query, active species, neighbors)."""
    z = np.arcsinh(library["delta"][:,active]/SCALE)
    if (indices.ndim != 3 or indices.shape[:2] != (len(query_base),int(active.sum()))
            or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or np.any(indices>=len(z))):
        raise ValueError("Invalid species neighbor graph")
    if mode == "correction":
        z -= np.arcsinh(library["base"][:,active]/SCALE)
    elif mode != "target":
        raise ValueError("Unknown transfer mode")
    # Match each neighbor's value to its species, not to all of its outputs.
    coordinate = z[indices,np.arange(z.shape[1])[None,:,None]].mean(axis=2)
    value = (decode(query_base[:,active],coordinate,dict(decoder="asinh",scale_floor=SCALE))
             if mode == "correction" else SCALE*np.sinh(coordinate))
    prediction = np.zeros_like(query_base)
    prediction[:,active] = value
    return prediction,coordinate


def score_transfers(staged,library,query,graphs,species,seed):
    populations = query["population"]
    active = np.array([s != "AR" for s in species])
    records = []
    for (space,size),indices in graphs.items():
        for k in (1,8):
            for mode in ("target","correction"):
                pred,coordinate = transfer_species(library,query["base"],indices[:,:,:k],active,mode)
                name = f"{space}_{size}-k{k}-{mode}.npz"
                np.savez(staged/name,prediction=pred,coordinate=coordinate,
                         source_indices=query["source_indices"],population=populations)
                for population in np.unique(populations):
                    mask = populations == population
                    q = {key:value[mask] for key,value in query.items()}
                    score = summarize(pred[mask],coordinate[mask],q,active,species,mode)
                    records.append(dict(seed=seed,space=space,library=size,k=k,mode=mode,
                                        shuffled=False,population=population,artifact=name,score=score))
    return records

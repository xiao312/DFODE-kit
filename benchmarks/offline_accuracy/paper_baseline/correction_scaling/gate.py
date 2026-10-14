"""Fail closed before a larger rung; never launch training from review code."""
import json
from benchmarks.flame_conditioning.extract import sha256
from .plan import SEEDS,SIZES,OBJECTIVES,configuration


def validate_record(record,size):
    previous = SIZES[SIZES.index(size)-1]
    if record.get("status")!="verified" or record.get("advance") is not True or record.get("size")!=previous:
        raise ValueError("Preceding rung must have a verified passing progression review")
    expected = {(s,o) for s in SEEDS for o in OBJECTIVES}
    keys = [(r["seed"],r["objective"]) for r in record.get("fits",[])]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError("Preceding review must include both objectives and both seeds")
    return previous


def require_previous(size,prepared,root,review_path,digest):
    if size==SIZES[0]:
        return None
    if root is None or review_path is None or not digest:
        raise ValueError("Larger rung requires --previous-root, --previous-review and --previous-review-sha256")
    if sha256(review_path)!=digest:
        raise ValueError("Previous review hash changed")
    record = json.loads(review_path.read_text())
    previous = validate_record(record,size)
    # Lazy imports: review consumes run's checked readers, never this gate at import time.
    from .review import recount,progression
    from .run import prepared_manifest
    for seed in SEEDS:
        manifest = prepared_manifest(prepared,seed)
        pair = []
        results = []
        for objective in OBJECTIVES:
            path = root/str(previous)/f"{seed}-{objective}"
            saved = next(r for r in record["fits"] if r["seed"]==seed and r["objective"]==objective)
            if sha256(path/"result.json")!=saved["result_sha256"]:
                raise ValueError("Previous result hash changed")
            summary,result = recount(path,prepared/str(seed),previous,manifest["species"])
            if summary!=saved or result["config"]!=configuration(seed,previous,objective):
                raise ValueError("Previous review scores or configuration do not reproduce")
            pair.append(summary)
            results.append(result)
        for key in ("initial_weights_sha256","prepared_manifest_sha256","training_count","row_presentations","source"):
            if results[0][key]!=results[1][key]:
                raise ValueError("Unpaired preceding rung: "+key)
        if not progression(*pair):
            raise ValueError("Preceding progression fails on independently recounted scores")
    return dict(size=previous,review_sha256=digest)

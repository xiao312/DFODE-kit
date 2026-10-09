"""Predeclared adaptations; do not choose seeds or tolerance from outcomes."""
NAMES = ("continue-coordinate", "finetune-physical", "finetune-tail",
         "relative-correction", "protected-correction", "local-state", "local-asinh",
         "arrhenius-heads", "arrhenius-lbfgs", "arrhenius-local")


def configuration(name, seed):
    if name not in NAMES or seed not in (20261011, 20261012):
        raise ValueError("Require a declared method and original seed")
    if name.startswith("arrhenius-"):
        result = dict(name=name, seed=seed, updates=8000 if name == "arrhenius-heads" else 0,
                    batch_size=256, widths=[32, 32], learning_rate=.001,
                    final_learning_rate=1e-5, physical_learning_rate=1e-4,
                    physical_final_learning_rate=1e-6, validation_every=1000,
                    wall_seconds=900, correction=False, local=name == "arrhenius-local",
                    tail_weight=0., guard_weight=0., neighbors=128, smoothing=1e-8,
                    atol=1e-15, rtol=.1, checkpoint_selection="final", input_floor=1e-30)
        if name == "arrhenius-lbfgs":
            result.update(updates=4080, adam_updates=4000, lbfgs_steps=80,
                          lbfgs_history=10, lbfgs_max_evaluations=400)
        return result
    correction = name.endswith("correction")
    return dict(name=name, seed=seed, updates=4000, batch_size=256,
                widths=[256]*4 if correction else [800]*4,
                learning_rate=1e-4 if correction else 1e-5,
                final_learning_rate=1e-6, validation_every=500, wall_seconds=900,
                correction=correction, local=name.startswith("local-"),
                tail_weight=.25 if name in ("finetune-tail", "protected-correction") else 0.,
                guard_weight=2. if name == "protected-correction" else 0.,
                neighbors=128, smoothing=1e-8, atol=1e-15, rtol=.1,
                checkpoint_selection="final")

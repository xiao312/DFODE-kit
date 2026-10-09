"""Predeclared matrix; never select a seed or tolerance after seeing results."""
TARGETS = ("state-boxcox", "gbct")
OBJECTIVES = ("coordinate", "increment", "state")
POLICIES = ("increment-reference-v1", "state-endpoint-v1")


def configuration(name, seed):
    matches = [(target, objective) for target in TARGETS for objective in OBJECTIVES
               if name == f"{target}-{objective}"]
    if len(matches) != 1 or seed not in (20261011, 20261012):
        raise ValueError("Select a declared target, objective and seed")
    target, objective = matches[0]
    return dict(name=name, target=target, objective=objective, seed=seed,
                widths=[800]*4, updates=4000, warmup=2000, batch_size=256,
                atol=1e-15, rtol=.1, interval=1e-6, lambda_a=.1, lambda_b=.5,
                learning_rate=.001, final_learning_rate=.00001,
                phase_two_learning_rate=.0001, phase_two_final_learning_rate=.000001,
                wall_seconds=900, validation_every=500, checkpoint_selection="final")

from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-stoich-interval",
        help="Evaluate a variable-dt stoichiometric interval checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mech", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--small-threshold",
        type=float,
        action="append",
        default=None,
        help="SSPI small-value threshold. Repeatable. Defaults to 1e-15 and 1e-12.",
    )


def handle_command(args):
    from dfode_kit.evaluation.stoich_interval import evaluate_stoich_interval_model

    results = evaluate_stoich_interval_model(
        args.checkpoint,
        args.source,
        args.output,
        mech_path=args.mech,
        device=args.device,
        small_thresholds=tuple(args.small_threshold or [1e-15, 1e-12]),
    )
    species = results["species_metrics"]["overall"]
    print(f"Saved interval evaluation metrics to {args.output}")
    print(
        "Species metrics: "
        f"mae={species['mae']:.6e}, "
        f"rmse={species['rmse']:.6e}, "
        f"r2={species['r2']:.6e}, "
        f"sspi_1e-12={species.get('sspi_1e-12')}"
    )
    print(
        "Interval metrics: "
        f"n_pairs={results['n_pairs']}, "
        f"dt_min={results['dt_min']:.6e}, "
        f"dt_max={results['dt_max']:.6e}, "
        f"temperature_mae={results['temperature_mae']:.6e}"
    )
    if "conservation_metrics" in results:
        conservation = results["conservation_metrics"]
        drift = results["update_mass_drift"]
        print(
            "Conservation metrics: "
            f"mean_mass_sum_error={conservation['mean_mass_sum_error']:.6e}, "
            f"negative_mass_fraction_rate={conservation['negative_mass_fraction_rate']:.6e}, "
            f"mean_abs_element_residual={conservation['mean_abs_element_residual']:.6e}, "
            f"mean_update_mass_drift={drift['mean_abs_sum_delta']:.6e}"
        )

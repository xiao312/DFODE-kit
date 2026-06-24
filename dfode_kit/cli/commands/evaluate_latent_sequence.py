from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-latent-sequence",
        help="Evaluate an AE + latent rollout checkpoint on a sequence validation split.",
    )
    parser.add_argument("--checkpoint", required=True, help="Torch checkpoint from train-latent-sequence.")
    parser.add_argument("--source", required=True, help="Input HDF5 sequence validation dataset.")
    parser.add_argument("--output", required=True, help="Output JSON metrics path.")
    parser.add_argument("--mech", default=None, help="Optional Cantera mechanism for hard-conservation diagnostics.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument(
        "--small-threshold",
        type=float,
        action="append",
        default=None,
        help="SSPI small-value threshold. Repeatable. Defaults to 1e-15 and 1e-12.",
    )


def handle_command(args):
    from dfode_kit.evaluation.latent_sequence import evaluate_latent_sequence_model

    results = evaluate_latent_sequence_model(
        args.checkpoint,
        args.source,
        args.output,
        device=args.device,
        small_thresholds=tuple(args.small_threshold or [1e-15, 1e-12]),
        mech_path=args.mech,
    )
    species = results["species_metrics"]["overall"]
    time = results["time_metrics"]
    latent_steps = results["latent_step_metrics"]
    print(f"Saved evaluation metrics to {args.output}")
    print(
        "Latent steps: "
        f"{latent_steps['steps_per_trajectory']} per trajectory, "
        f"{latent_steps['total_latent_transitions']} total transitions, "
        f"mean_true_horizon={latent_steps['mean_true_horizon']:.6e}, "
        f"mean_predicted_horizon={latent_steps['mean_predicted_horizon']:.6e}"
    )
    print(
        "Species metrics: "
        f"mae={species['mae']:.6e}, rmse={species['rmse']:.6e}, "
        f"r2={species['r2']:.6e}, sspi_1e-12={species.get('sspi_1e-12')}"
    )
    print(
        "Time metrics: "
        f"mae_log_dt={time['mae_log_dt']:.6e}, "
        f"median_relative_dt_error={time['median_relative_dt_error']:.6e}"
    )
    if "conservation_metrics" in results:
        conservation = results["conservation_metrics"]
        completion = conservation["completion_matrix"]
        print(
            "Conservation metrics: "
            f"mean_mass_sum_error={conservation['mean_mass_sum_error']:.6e}, "
            f"negative_mass_fraction_rate={conservation['negative_mass_fraction_rate']:.6e}, "
            f"mean_abs_element_residual={conservation['mean_abs_element_residual']:.6e}, "
            f"key_species={completion['n_key_species']}/{completion['n_species']}"
        )

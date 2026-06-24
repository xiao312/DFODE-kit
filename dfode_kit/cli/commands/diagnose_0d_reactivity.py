from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "diagnose-0d-reactivity",
        help="Summarize whether 0D sequence datasets contain enough reactive states.",
    )
    parser.add_argument("--input", required=True, help="Input sequence-v2 HDF5 file.")
    parser.add_argument("--output", required=True, help="Output JSON summary path.")
    parser.add_argument("--step-temperature-delta", type=float, default=1.0)
    parser.add_argument("--step-species-delta", type=float, default=1e-6)
    parser.add_argument("--trajectory-temperature-rise", type=float, default=50.0)
    parser.add_argument("--reactive-step-fraction-threshold", type=float, default=0.05)


def handle_command(args):
    from dfode_kit.data.reactivity import write_reactivity_summary

    summary = write_reactivity_summary(
        args.input,
        args.output,
        step_temperature_delta=args.step_temperature_delta,
        step_species_delta=args.step_species_delta,
        trajectory_temperature_rise=args.trajectory_temperature_rise,
        reactive_step_fraction_threshold=args.reactive_step_fraction_threshold,
    )
    aggregate = summary["aggregate"]
    print(f"Saved reactivity summary to {args.output}")
    print(
        "Reactivity: "
        f"trajectories={aggregate['n_reactive_trajectories']}/{aggregate['n_trajectories']}, "
        f"reactive_fraction={aggregate['reactive_trajectory_fraction']:.3f}, "
        f"mean_reactive_state_fraction={aggregate['mean_reactive_state_fraction']:.3f}, "
        f"max_dT={aggregate['max_temperature_rise']:.3f} K"
    )

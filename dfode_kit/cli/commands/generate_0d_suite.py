from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-0d-suite",
        help="Generate train/validation 0D reactor sequence suites across mechanisms and operating conditions.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for HDF5 split files and manifest.json.")
    parser.add_argument(
        "--profile",
        choices=("pilot", "small", "reactive-pilot", "crossfuel-packaged-pilot"),
        default="pilot",
        help="Operating-condition grid profile.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of operating conditions held out for validation.")
    parser.add_argument("--seed", type=int, default=260624, help="Deterministic split seed.")
    parser.add_argument("--skip-reactivity-diagnostics", action="store_true", help="Do not write reactivity summaries into manifest.")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to case_id or mechanism_id. Repeatable.",
    )


def handle_command(args):
    from dfode_kit.data.sequence_suites import generate_suite

    manifest = generate_suite(
        args.output_dir,
        profile=args.profile,
        val_fraction=args.val_fraction,
        seed=args.seed,
        only=set(args.only or []) or None,
        diagnose_reactivity=not args.skip_reactivity_diagnostics,
    )

    print(f"Saved sequence suite to {args.output_dir}")
    for case in manifest["cases"]:
        print(
            f"{case['case_id']}: "
            f"train={case['n_train']} val={case['n_val']} "
            f"steps={case['steps']} sampling={case['sampling']}"
        )
        for split_name, split in case["splits"].items():
            reactivity = split.get("reactivity")
            if reactivity is not None:
                print(
                    f"  {split_name}: reactive_trajectories="
                    f"{reactivity['n_reactive_trajectories']}/{reactivity['n_trajectories']}, "
                    f"mean_reactive_state_fraction={reactivity['mean_reactive_state_fraction']:.3f}, "
                    f"max_dT={reactivity['max_temperature_rise']:.1f} K"
                )

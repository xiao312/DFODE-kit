from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-hybrid-integrators",
        help="Compare adaptive BE and SDIRK2 with hold, Euler, and neural implicit-root guesses.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--phase-name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=20260713)
    parser.add_argument("--max-accepted-steps", type=int, default=256)
    parser.add_argument("--min-substep", type=float, default=1e-12)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--species-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--temperature-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--newton-tolerance", type=float, default=1.0)
    parser.add_argument("--max-newton-iterations", type=int, default=12)
    parser.add_argument("--max-line-search-steps", type=int, default=12)
    parser.add_argument("--fd-relative-step", type=float, default=1e-5)
    parser.add_argument("--fd-species-scale", type=float, default=1e-5)
    parser.add_argument("--minimum-temperature", type=float, default=200.0)
    parser.add_argument("--positivity-floor", type=float, default=0.0)
    parser.add_argument("--sdirk-error-tolerance", type=float, default=1.0)
    parser.set_defaults(command_handler=handle_command)


def handle_command(args):
    from dfode_kit.evaluation.hybrid_integrators import evaluate_hybrid_integrators

    result = evaluate_hybrid_integrators(
        args.checkpoint,
        args.source,
        args.mechanism,
        args.output,
        phase_name=args.phase_name,
        device=args.device,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        max_accepted_steps=args.max_accepted_steps,
        min_substep=args.min_substep,
        relative_tolerance=args.relative_tolerance,
        species_absolute_tolerance=args.species_absolute_tolerance,
        temperature_absolute_tolerance=args.temperature_absolute_tolerance,
        newton_tolerance=args.newton_tolerance,
        max_newton_iterations=args.max_newton_iterations,
        max_line_search_steps=args.max_line_search_steps,
        fd_relative_step=args.fd_relative_step,
        fd_species_scale=args.fd_species_scale,
        minimum_temperature=args.minimum_temperature,
        positivity_floor=args.positivity_floor,
        sdirk_error_tolerance=args.sdirk_error_tolerance,
    )
    print(f"wrote hybrid-integrator report: {args.output}")
    print(f"wrote per-sample report: {result['sample_csv']}")
    for name, metrics in result["method_metrics"].items():
        print(
            f"{name}: completed={metrics['completion_rate']:.3f}, "
            f"steps={metrics['accepted_steps']['mean']}, "
            f"newton={metrics['newton_iterations']['mean']}, "
            f"rhs={metrics['rhs_evaluations']['mean']}, "
            f"species_mae={metrics['species_mae_completed']['mean']}"
        )


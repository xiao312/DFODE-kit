from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "benchmark-stoich-interval-runtime",
        help="Benchmark variable-dt interval checkpoint inference runtime.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mech", default=None, help="Cantera mechanism for optional CVODE timing.")
    parser.add_argument("--cvode-samples", type=int, default=0, help="Number of samples to time with Cantera/CVODE.")


def handle_command(args):
    from dfode_kit.evaluation.runtime_benchmark import benchmark_stoich_interval_runtime

    result = benchmark_stoich_interval_runtime(
        args.checkpoint,
        args.source,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        repeat=args.repeat,
        warmup=args.warmup,
        max_samples=args.max_samples or None,
        mech_path=args.mech,
        cvode_samples=args.cvode_samples,
    )
    print(f"Saved runtime benchmark to {args.output}")
    print(
        "Runtime: "
        f"n_samples={result['n_samples']}, "
        f"mean_seconds={result['mean_seconds']:.6e}, "
        f"microseconds_per_sample={result['microseconds_per_sample']:.6e}"
    )
    if "cvode" in result:
        cvode = result["cvode"]
        print(
            "CVODE: "
            f"n_samples={cvode['n_samples']}, "
            f"microseconds_per_sample={cvode['microseconds_per_sample']:.6e}, "
            f"neural_speedup_vs_cvode={cvode['neural_speedup_vs_cvode']:.6e}"
        )

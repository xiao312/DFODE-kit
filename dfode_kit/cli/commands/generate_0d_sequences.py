from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-0d-sequences",
        help="Generate 0D constant-pressure reactor trajectory datasets.",
    )
    parser.add_argument("--mech", required=True, help="Path to the Cantera YAML mechanism.")
    parser.add_argument("--output", required=True, help="Output HDF5 sequence dataset path.")
    parser.add_argument("--fuel", required=True, help="Fuel composition, e.g. 'H2:1.0'.")
    parser.add_argument("--oxidizer", default="O2:1.0,N2:3.76", help="Oxidizer composition.")
    parser.add_argument("--temperature", type=float, action="append", required=True, help="Initial temperature in K. Repeatable.")
    parser.add_argument("--pressure", type=float, action="append", help="Initial pressure in Pa. Repeatable.")
    parser.add_argument("--phi", type=float, action="append", help="Equivalence ratio. Repeatable.")
    parser.add_argument("--dt", type=float, default=1e-6, help="Time step between saved states in seconds.")
    parser.add_argument("--steps", type=int, default=100, help="Number of saved integration steps.")
    parser.add_argument("--sampling", choices=("fixed", "log-time"), default="fixed", help="Saved-time sampling strategy.")
    parser.add_argument("--t-end", type=float, default=None, help="End time for log-time sampling.")
    parser.add_argument("--t-start", type=float, default=None, help="First nonzero time for log-time sampling.")
    parser.add_argument("--mechanism-id", default=None, help="Stable mechanism identifier stored in metadata.")
    parser.add_argument("--energy", choices=("on", "off"), default="on", help="Cantera reactor energy mode.")


def handle_command(args):
    from dfode_kit.data.sequences import build_reactor_conditions, log_time_grid, write_sequence_dataset

    conditions = build_reactor_conditions(
        temperatures=args.temperature,
        pressures=args.pressure or [101325.0],
        phis=args.phi or [1.0],
        fuel=args.fuel,
        oxidizer=args.oxidizer,
    )
    times = None
    dt = args.dt
    if args.sampling == "log-time":
        if args.t_end is None:
            raise ValueError("--t-end is required for log-time sampling")
        times = log_time_grid(t_end=args.t_end, steps=args.steps, t_start=args.t_start)
        dt = None

    write_sequence_dataset(
        args.output,
        args.mech,
        conditions,
        dt=dt,
        steps=args.steps,
        times=times,
        energy=args.energy,
        mechanism_id=args.mechanism_id,
        sampling=args.sampling,
    )
    print(f"Saved {len(conditions)} 0D sequence(s) to {args.output}")

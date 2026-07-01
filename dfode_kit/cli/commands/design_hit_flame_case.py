from __future__ import annotations

import json


def _parse_csv_floats(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    return tuple(float(part) for part in text.split(",") if part.strip())


def _parse_csv_ints(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    return tuple(int(part) for part in text.split(",") if part.strip())


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "design-hit-flame-case",
        help="Design a mechanism-aware 2D HIT premixed-flame validation case.",
    )
    parser.add_argument("--case-id", required=True, help="Stable case identifier used for the output subdirectory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the case design bundle will be written.")
    parser.add_argument("--mechanism", required=True, help="Cantera mechanism YAML path/name.")
    parser.add_argument("--fuel", required=True, help="Fuel composition passed to Cantera set_equivalence_ratio.")
    parser.add_argument("--oxidizer", default="O2:1.0,N2:3.76", help="Oxidizer composition.")
    parser.add_argument("--phi", type=float, required=True, help="Equivalence ratio.")
    parser.add_argument("--Tu", type=float, required=True, help="Unburned temperature in K.")
    parser.add_argument("--Pu", type=float, required=True, help="Unburned pressure in Pa.")
    parser.add_argument("--lt-over-deltaL", type=float, required=True, help="Target integral length scale divided by laminar flame thickness.")
    parser.add_argument("--uprime-over-Sl", type=float, required=True, help="Target turbulence intensity divided by laminar flame speed.")
    parser.add_argument("--Sl", type=float, default=None, help="Optional laminar flame speed in m/s. If omitted, solve a Cantera flame.")
    parser.add_argument("--deltaL", type=float, default=None, help="Optional laminar flame thickness in m. If omitted, use Cantera delta_alpha.")
    parser.add_argument("--flame-width", type=float, default=0.03, help="Cantera FreeFlame domain width in m.")
    parser.add_argument("--transport-model", default="mixture-averaged", help="Cantera transport model for flame-property solve.")
    parser.add_argument("--loglevel", type=int, default=0, help="Cantera flame solver loglevel.")
    parser.add_argument("--flame-radius", type=float, default=None, help="Optional intended initial flame/kernel radius in m, recorded as metadata.")
    parser.add_argument("--kernel-temperature", type=float, default=None, help="Optional intended hot-kernel temperature in K, recorded as metadata.")
    parser.add_argument("--scalar-initialization-note", default=None, help="Free-form note about intended T/Y_i/pressure initialization.")
    parser.add_argument("--N", type=int, default=128, help="Default grid size for HIT tuning.")
    parser.add_argument("--N-values", default=None, help="Optional comma-separated grid-size search list.")
    parser.add_argument("--L", type=float, default=6.283185307179586, help="Default physical domain length for HIT tuning.")
    parser.add_argument("--L-values", default=None, help="Optional comma-separated domain-length search list.")
    parser.add_argument("--dt", type=float, default=0.05, help="NS2dLab evolution time step used during optional tuning evolution.")
    parser.add_argument("--nu", type=float, default=1e-4, help="Kinematic viscosity used during optional tuning evolution.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help="NS2dLab backend.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for spectral field generation.")
    parser.add_argument("--evolve-seconds", type=float, default=0.0, help="Optionally evolve candidates and select the best sampled field.")
    parser.add_argument("--sample-every", type=int, default=1, help="Candidate sampling interval during optional evolution.")
    parser.add_argument("--k0-values", default="2,3,4,5,6,8,10,12,16", help="Comma-separated spectral peak search values.")
    parser.add_argument("--mode-values", default=None, help="Optional comma-separated modal indices; overrides physical k0 search.")
    parser.add_argument("--bandwidth", type=float, default=1.5, help="Physical-k0 Gaussian bandwidth.")
    parser.add_argument("--bandwidth-modes", type=float, default=1.5, help="Modal-index Gaussian bandwidth.")
    parser.add_argument("--write-search-plot", action="store_true", help="Write requested-vs-achieved regime search figure.")
    parser.add_argument("--write-artifacts", action="store_true", help="Write NS2dLab curl/history diagnostic artifacts.")


def handle_command(args):
    from dfode_kit.cfd_init.hit2d import FlameCaseDesign, HitTuningConfig, design_hit_flame_case

    if (args.Sl is None) != (args.deltaL is None):
        raise ValueError("--Sl and --deltaL must be provided together")

    payload = design_hit_flame_case(
        FlameCaseDesign(
            case_id=args.case_id,
            mechanism=args.mechanism,
            fuel=args.fuel,
            oxidizer=args.oxidizer,
            phi=args.phi,
            T_u=args.Tu,
            P_u=args.Pu,
            target_lt_over_deltaL=args.lt_over_deltaL,
            target_uprime_over_Sl=args.uprime_over_Sl,
            flame_radius=args.flame_radius,
            kernel_temperature=args.kernel_temperature,
            scalar_initialization_note=args.scalar_initialization_note,
        ),
        HitTuningConfig(
            N=args.N,
            L=args.L,
            dt=args.dt,
            nu=args.nu,
            backend=args.backend,
            seed=args.seed,
            evolve_seconds=args.evolve_seconds,
            sample_every=args.sample_every,
            N_values=_parse_csv_ints(args.N_values),
            L_values=_parse_csv_floats(args.L_values),
            k0_values=_parse_csv_floats(args.k0_values) or (),
            mode_values=_parse_csv_floats(args.mode_values),
            bandwidth=args.bandwidth,
            bandwidth_modes=args.bandwidth_modes,
        ),
        output_dir=args.output_dir,
        S_L=args.Sl,
        delta_L=args.deltaL,
        flame_width=args.flame_width,
        transport_model=args.transport_model,
        loglevel=args.loglevel,
        write_search_plot=args.write_search_plot,
        write_artifacts=args.write_artifacts,
    )
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))

from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "train-positive-interval",
        help="Train positivity-preserving Patankar or reaction-trajectory interval models.",
    )
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--mech", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("neural-patankar", "reaction-trajectory"), required=True)
    parser.add_argument("--phase-name")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--delta-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--source-loss-weight",
        type=float,
        default=0.0,
        help="Weight for MAE(delta_Y / dt) against Fluent native-DI labels.",
    )
    parser.add_argument(
        "--source-loss-scale",
        type=float,
        default=1.0,
        help="Fixed physical scale for source MAE in 1/s; never batch-derived.",
    )
    parser.add_argument(
        "--mixture-molecular-weight-loss-weight",
        type=float,
        default=0.0,
        help="Weight for endpoint mixture-molecular-weight MAE.",
    )
    parser.add_argument(
        "--mixture-molecular-weight-loss-scale",
        type=float,
        default=1.0,
        help="Fixed molecular-weight MAE scale in kg/kmol; never batch-derived.",
    )
    parser.add_argument(
        "--density-increment-loss-weight",
        type=float,
        default=0.0,
        help="Weight for ideal-gas density-increment MAE.",
    )
    parser.add_argument(
        "--density-increment-loss-scale",
        type=float,
        default=1.0,
        help="Fixed density-increment MAE scale in kg/m^3; never batch-derived.",
    )
    parser.add_argument(
        "--jvp-dataset",
        help=(
            "Optional fluent-di-jvp-v1 paired native-DI dataset. Supplying it "
            "enables deterministic train/validation source-JVP reporting."
        ),
    )
    parser.add_argument(
        "--source-jvp-loss-weight",
        type=float,
        default=0.0,
        help="Weight for native-DI source-map directional-response MAE.",
    )
    parser.add_argument(
        "--source-jvp-loss-scale",
        type=float,
        default=1.0,
        help="Fixed physical source-JVP MAE scale; never batch-derived.",
    )
    parser.add_argument(
        "--thermochemical-output-mode",
        choices=("delta-temperature", "delta-h-total"),
        default="delta-temperature",
        help=(
            "Predict legacy delta_T or predict Fluent total-enthalpy increment "
            "and delegate temperature recovery to Fluent."
        ),
    )
    parser.add_argument(
        "--total-enthalpy-delta-scale",
        type=float,
        default=1.0e3,
        help="Fixed delta_h_total target/output scale in J/kg.",
    )
    parser.add_argument(
        "--total-enthalpy-delta-loss-weight",
        type=float,
        default=1.0,
        help="Weight for MAE in fixed-scale normalized delta_h_total space.",
    )
    parser.add_argument(
        "--total-enthalpy-target-transform",
        choices=("identity", "signed-power"),
        default="identity",
    )
    parser.add_argument(
        "--total-enthalpy-transform-alpha", type=float, default=0.1
    )
    parser.add_argument(
        "--total-enthalpy-transformed-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--extent-scale", type=float, default=1e-4)
    parser.add_argument(
        "--availability-mode",
        choices=("hard", "smooth-safe"),
        default="hard",
        help="Reactant-availability allocation used by NeuralPatankar.",
    )
    parser.add_argument(
        "--availability-p-norm",
        type=float,
        default=32.0,
        help="Generalized harmonic-minimum order for smooth-safe allocation.",
    )
    parser.add_argument("--trajectory-scale", type=float, default=1e-5)
    parser.add_argument("--mobility-scale", type=float, default=1e-4)
    parser.add_argument("--proximal-steps", type=int, default=3)
    parser.add_argument("--proximal-beta", type=float, default=1.0)
    parser.add_argument("--positivity-floor", type=float, default=1e-30)
    parser.add_argument("--seed", type=int, default=260624)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def handle_command(args):
    from dfode_kit.training.positive_interval import (
        PositiveIntervalTrainingConfig,
        train_positive_interval_model,
    )

    config = PositiveIntervalTrainingConfig(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        transform_alpha=args.transform_alpha,
        delta_loss_weight=args.delta_loss_weight,
        source_loss_weight=args.source_loss_weight,
        source_loss_scale=args.source_loss_scale,
        mixture_molecular_weight_loss_weight=args.mixture_molecular_weight_loss_weight,
        mixture_molecular_weight_loss_scale=args.mixture_molecular_weight_loss_scale,
        density_increment_loss_weight=args.density_increment_loss_weight,
        density_increment_loss_scale=args.density_increment_loss_scale,
        jvp_dataset=args.jvp_dataset,
        source_jvp_loss_weight=args.source_jvp_loss_weight,
        source_jvp_loss_scale=args.source_jvp_loss_scale,
        thermochemical_output_mode=args.thermochemical_output_mode,
        total_enthalpy_delta_scale=args.total_enthalpy_delta_scale,
        total_enthalpy_delta_loss_weight=args.total_enthalpy_delta_loss_weight,
        total_enthalpy_target_transform=(
            args.total_enthalpy_target_transform
        ),
        total_enthalpy_transform_alpha=args.total_enthalpy_transform_alpha,
        total_enthalpy_transformed_loss_weight=(
            args.total_enthalpy_transformed_loss_weight
        ),
        extent_scale=args.extent_scale,
        availability_mode=args.availability_mode,
        availability_p_norm=args.availability_p_norm,
        trajectory_scale=args.trajectory_scale,
        mobility_scale=args.mobility_scale,
        proximal_steps=args.proximal_steps,
        proximal_beta=args.proximal_beta,
        positivity_floor=args.positivity_floor,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        log_every=args.log_every,
    )
    metrics = train_positive_interval_model(
        args.train_source,
        args.validation_source,
        args.mech,
        args.output,
        phase_name=args.phase_name,
        device=args.device,
        config=config,
    )
    print(metrics)

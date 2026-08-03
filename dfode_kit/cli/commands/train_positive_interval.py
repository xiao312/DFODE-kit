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
    parser.add_argument("--extent-scale", type=float, default=1e-4)
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
        extent_scale=args.extent_scale,
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

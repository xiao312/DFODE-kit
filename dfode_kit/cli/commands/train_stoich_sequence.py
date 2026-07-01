from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "train-stoich-sequence",
        help="Train a hard-conserved stoichiometric reaction-flux baseline.",
    )
    parser.add_argument("--source", required=True, help="Input HDF5 sequence dataset.")
    parser.add_argument("--output", required=True, help="Output Torch checkpoint path.")
    parser.add_argument("--mech", required=True, help="Cantera mechanism used for stoichiometric fluxes.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-weight", type=float, default=0.01)
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--species-loss-weight", type=float, default=0.1)
    parser.add_argument("--loss-kind", choices=("mse", "mae"), default="mae")
    parser.add_argument("--flux-mode", choices=("direct", "signed-power"), default="signed-power")
    parser.add_argument(
        "--no-transform-scale-by-alpha",
        action="store_false",
        dest="transform_scale_by_alpha",
        default=True,
        help="Disable Ke-style /alpha scaling for ablations.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cpu' or 'cuda:0'.")


def handle_command(args):
    from dfode_kit.training.stoich_sequence import StoichSequenceTrainingConfig, train_stoich_sequence_model

    metrics = train_stoich_sequence_model(
        args.source,
        args.output,
        args.mech,
        config=StoichSequenceTrainingConfig(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            time_weight=args.time_weight,
            transform_alpha=args.transform_alpha,
            species_loss_weight=args.species_loss_weight,
            loss_kind=args.loss_kind,
            transform_scale_by_alpha=args.transform_scale_by_alpha,
            flux_mode=args.flux_mode,
        ),
        device=args.device,
    )
    print(f"Saved stoichiometric sequence model to {args.output}")
    print(
        "Final losses: "
        f"loss={metrics['loss']:.6e}, "
        f"state={metrics['state_loss']:.6e}, "
        f"time={metrics['time_loss']:.6e}"
    )

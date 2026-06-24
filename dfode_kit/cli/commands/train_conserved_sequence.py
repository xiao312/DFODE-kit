from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "train-conserved-sequence",
        help="Train a hard atom-conserving key-species delta baseline.",
    )
    parser.add_argument("--source", required=True, help="Input HDF5 sequence dataset.")
    parser.add_argument("--output", required=True, help="Output Torch checkpoint path.")
    parser.add_argument("--mech", required=True, help="Cantera mechanism used for atom conservation.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-weight", type=float, default=0.01)
    parser.add_argument(
        "--key-mode",
        choices=("direct", "signed-power", "formation-consumption"),
        default="signed-power",
        help="Key-delta parameterization before the hard completion layer.",
    )
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--transform-eps", type=float, default=1e-12)
    parser.add_argument(
        "--transform-scale-by-alpha",
        action="store_true",
        default=True,
        help="Use Ke-style sign(x)*abs(x)^alpha/alpha scaling for signed-power transforms.",
    )
    parser.add_argument(
        "--no-transform-scale-by-alpha",
        action="store_false",
        dest="transform_scale_by_alpha",
        help="Disable Ke-style /alpha scaling for ablations.",
    )
    parser.add_argument(
        "--key-scale-alpha",
        type=float,
        default=0.0,
        help="For formation-consumption mode, multiply net key delta by (abs(Y_key)+eps)^alpha.",
    )
    parser.add_argument(
        "--key-loss-weight",
        type=float,
        default=0.1,
        help="Auxiliary key-space target loss weight.",
    )
    parser.add_argument(
        "--species-loss-mode",
        choices=(
            "physical",
            "signed-power",
            "signed-power-delta",
            "physical-plus-signed-power",
            "signed-power-plus-delta",
        ),
        default="signed-power-plus-delta",
        help="Species loss space. Signed-power emphasizes small species without changing the hard update.",
    )
    parser.add_argument("--species-transform-alpha", type=float, default=0.1)
    parser.add_argument(
        "--species-loss-weight",
        type=float,
        default=0.1,
        help="Weight for transformed species loss in physical-plus-signed-power mode.",
    )
    parser.add_argument(
        "--loss-kind",
        choices=("mse", "mae"),
        default="mae",
        help="Regression loss kind for state, transformed species, key, and time losses.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cpu' or 'cuda:0'.")


def handle_command(args):
    from dfode_kit.training.conserved_sequence import (
        ConservedSequenceTrainingConfig,
        train_conserved_sequence_model,
    )

    metrics = train_conserved_sequence_model(
        args.source,
        args.output,
        args.mech,
        config=ConservedSequenceTrainingConfig(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            time_weight=args.time_weight,
            key_mode=args.key_mode,
            transform_alpha=args.transform_alpha,
            transform_eps=args.transform_eps,
            transform_scale_by_alpha=args.transform_scale_by_alpha,
            key_scale_alpha=args.key_scale_alpha,
            key_loss_weight=args.key_loss_weight,
            species_loss_mode=args.species_loss_mode,
            species_transform_alpha=args.species_transform_alpha,
            species_loss_weight=args.species_loss_weight,
            loss_kind=args.loss_kind,
        ),
        device=args.device,
    )
    print(f"Saved conserved sequence model to {args.output}")
    print(
        "Final losses: "
        f"loss={metrics['loss']:.6e}, "
        f"state={metrics['state_loss']:.6e}, "
        f"time={metrics['time_loss']:.6e}, "
        f"key={metrics['key_loss']:.6e}"
    )

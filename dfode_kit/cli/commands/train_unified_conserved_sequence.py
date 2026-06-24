from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "train-unified-conserved-sequence",
        help="Train a shared-latent, mechanism-specific hard-conserved sequence model from a suite manifest.",
    )
    parser.add_argument("--manifest", required=True, help="Input suite manifest.json.")
    parser.add_argument("--output", required=True, help="Output Torch checkpoint path.")
    parser.add_argument("--eval-output", required=True, help="Output JSON validation summary.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time-weight", type=float, default=0.01)
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--key-loss-weight", type=float, default=0.1)
    parser.add_argument("--species-loss-weight", type=float, default=0.1)
    parser.add_argument("--loss-kind", choices=("mse", "mae"), default="mae")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cpu' or 'cuda:0'.")


def handle_command(args):
    from dfode_kit.training.unified_conserved_sequence import (
        UnifiedConservedTrainingConfig,
        train_unified_conserved_sequence_model,
    )

    results = train_unified_conserved_sequence_model(
        args.manifest,
        args.output,
        args.eval_output,
        config=UnifiedConservedTrainingConfig(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            time_weight=args.time_weight,
            transform_alpha=args.transform_alpha,
            key_loss_weight=args.key_loss_weight,
            species_loss_weight=args.species_loss_weight,
            loss_kind=args.loss_kind,
        ),
        device=args.device,
    )
    print(f"Saved unified conserved sequence model to {args.output}")
    print(f"Saved validation summary to {args.eval_output}")
    print(f"Final loss: {results['final_metrics']['loss']:.6e}")
    for name, group in results["groups"].items():
        species = group["species_metrics"]["overall"]
        conservation = group["conservation_metrics"]
        print(
            f"{name}: "
            f"mae={species['mae']:.6e}, r2={species['r2']:.6e}, "
            f"sspi_1e-12={species.get('sspi_1e-12')}, "
            f"neg={conservation['negative_mass_fraction_rate']:.6e}, "
            f"mass={conservation['mean_mass_sum_error']:.6e}"
        )

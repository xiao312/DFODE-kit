from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "train-latent-sequence",
        help="Train a minimal AE + GRU latent rollout baseline on 0D sequence data.",
    )
    parser.add_argument("--source", required=True, help="Input HDF5 sequence dataset.")
    parser.add_argument("--output", required=True, help="Output Torch checkpoint path.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--time-weight", type=float, default=0.01)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cpu' or 'cuda:0'.")


def handle_command(args):
    from dfode_kit.training.latent_sequence import LatentSequenceTrainingConfig, train_latent_sequence_model

    metrics = train_latent_sequence_model(
        args.source,
        args.output,
        config=LatentSequenceTrainingConfig(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            reconstruction_weight=args.reconstruction_weight,
            time_weight=args.time_weight,
        ),
        device=args.device,
    )
    print(f"Saved latent sequence model to {args.output}")
    print(
        "Final losses: "
        f"loss={metrics['loss']:.6e}, "
        f"rollout={metrics['rollout_loss']:.6e}, "
        f"reconstruction={metrics['reconstruction_loss']:.6e}, "
        f"time={metrics['time_loss']:.6e}"
    )

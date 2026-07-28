from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "train-stoich-interval",
        help="Train a variable-dt stoichiometric interval baseline.",
    )
    parser.add_argument("--source", required=True, help="Input interval-pair HDF5 dataset.")
    parser.add_argument("--output", required=True, help="Output Torch checkpoint path.")
    parser.add_argument("--mech", required=True, help="Cantera mechanism used for stoichiometric fluxes.")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260728,
        help="Global model-initialization and epoch-shuffle seed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic Torch algorithms and disable TF32.",
    )
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--species-loss-weight", type=float, default=0.1)
    parser.add_argument("--loss-kind", choices=("mse", "mae"), default="mae")
    parser.add_argument("--flux-mode", choices=("direct", "signed-power"), default="signed-power")
    parser.add_argument(
        "--model-variant",
        choices=("stoich", "thermo-affinity", "substep", "substep-soft-thermo", "thermo-progress-substep"),
        default="stoich",
    )
    parser.add_argument("--thermo-extent-scale", type=float, default=1e-3)
    parser.add_argument("--thermo-force-scale", type=float, default=1.0)
    parser.add_argument("--thermo-availability-threshold", type=float, default=-60.0)
    parser.add_argument("--thermo-availability-slope", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--adaptive-substeps", action="store_true")
    parser.add_argument("--semigroup-loss-weight", type=float, default=0.0)
    parser.add_argument("--midpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--thermo-embedding-dim", type=int, default=32)
    parser.add_argument("--log-every-epochs", type=int, default=50)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--reaction-extent-raw-clip", type=float, default=0.0)
    parser.add_argument("--thermo-feature-clip", type=float, default=10.0)
    parser.add_argument("--error-loss-weight", type=float, default=0.0)
    parser.add_argument("--step-policy-loss-weight", type=float, default=0.0)
    parser.add_argument("--learned-adaptive-substeps", action="store_true")
    parser.add_argument("--thermo-progress-force-clip", type=float, default=8.0)
    parser.add_argument("--thermo-progress-mobility-scale", type=float, default=1.0)
    parser.add_argument(
        "--thermo-progress-mode",
        choices=("sinh", "linear-affinity", "monotone-sinh", "monotone-tanh"),
        default="sinh",
    )
    parser.add_argument("--latent-update-mode", choices=("residual", "bounded"), default="residual")
    parser.add_argument("--latent-step-scale", type=float, default=1.0)
    parser.add_argument("--proximal-correction-steps", type=int, default=0)
    parser.add_argument("--proximal-correction-lr", type=float, default=0.1)
    parser.add_argument("--proximal-free-energy-weight", type=float, default=0.0)
    parser.add_argument("--proximal-y-floor", type=float, default=1e-300)
    parser.add_argument("--positivity-safety-factor", type=float, default=0.0)
    parser.add_argument(
        "--no-skip-nonfinite-batches",
        action="store_false",
        dest="skip_nonfinite_batches",
        default=True,
    )
    parser.add_argument(
        "--input-perturb-alpha",
        type=float,
        default=0.0,
        help="Training-time denoising perturbation strength for current input states.",
    )
    parser.add_argument("--input-perturb-seed", type=int, default=20260626)
    parser.add_argument(
        "--no-transform-scale-by-alpha",
        action="store_false",
        dest="transform_scale_by_alpha",
        default=True,
        help="Disable Ke-style /alpha scaling for ablations.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cpu' or 'cuda:0'.")


def handle_command(args):
    from dfode_kit.training.stoich_interval import StoichIntervalTrainingConfig, train_stoich_interval_model

    metrics = train_stoich_interval_model(
        args.source,
        args.output,
        args.mech,
        config=StoichIntervalTrainingConfig(
            seed=args.seed,
            deterministic=args.deterministic,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            transform_alpha=args.transform_alpha,
            species_loss_weight=args.species_loss_weight,
            loss_kind=args.loss_kind,
            transform_scale_by_alpha=args.transform_scale_by_alpha,
            flux_mode=args.flux_mode,
            input_perturb_alpha=args.input_perturb_alpha,
            input_perturb_seed=args.input_perturb_seed,
            model_variant=args.model_variant,
            thermo_extent_scale=args.thermo_extent_scale,
            thermo_force_scale=args.thermo_force_scale,
            thermo_availability_threshold=args.thermo_availability_threshold,
            thermo_availability_slope=args.thermo_availability_slope,
            substeps=args.substeps,
            adaptive_substeps=args.adaptive_substeps,
            semigroup_loss_weight=args.semigroup_loss_weight,
            midpoint_loss_weight=args.midpoint_loss_weight,
            thermo_embedding_dim=args.thermo_embedding_dim,
            log_every_epochs=args.log_every_epochs,
            gradient_clip_norm=args.gradient_clip_norm,
            reaction_extent_raw_clip=args.reaction_extent_raw_clip,
            thermo_feature_clip=args.thermo_feature_clip,
            skip_nonfinite_batches=args.skip_nonfinite_batches,
            error_loss_weight=args.error_loss_weight,
            step_policy_loss_weight=args.step_policy_loss_weight,
            learned_adaptive_substeps=args.learned_adaptive_substeps,
            thermo_progress_force_clip=args.thermo_progress_force_clip,
            thermo_progress_mobility_scale=args.thermo_progress_mobility_scale,
            thermo_progress_mode=args.thermo_progress_mode,
            latent_update_mode=args.latent_update_mode,
            latent_step_scale=args.latent_step_scale,
            proximal_correction_steps=args.proximal_correction_steps,
            proximal_correction_lr=args.proximal_correction_lr,
            proximal_free_energy_weight=args.proximal_free_energy_weight,
            proximal_y_floor=args.proximal_y_floor,
            positivity_safety_factor=args.positivity_safety_factor,
        ),
        device=args.device,
    )
    print(f"Saved stoichiometric interval model to {args.output}")
    print(
        "Final losses: "
        f"loss={metrics['loss']:.6e}, "
        f"state={metrics['state_loss']:.6e}, "
        f"tp={metrics['tp_loss']:.6e}, "
        f"y={metrics['y_loss']:.6e}, "
        f"delta={metrics['delta_loss']:.6e}"
    )

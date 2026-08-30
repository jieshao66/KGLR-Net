#!/usr/bin/env python3
"""Train KGLR-Net with one fixed-shot annotated training split.

The command trains GPPS and then LPRS. It never constructs an evaluation data
loader and never uses held-out samples for optimization or checkpoint choice.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from kglr_net.data import KGLRDataset, SUPPORTED_DATASETS
from kglr_net.training import (
    TrainingConfig,
    make_data_generator,
    seed_data_worker,
    train_two_stage,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train GPPS and LPRS from one annotated 5-, 10-, or 20-shot split.")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing <dataset>/<train|test>/npz_data.",
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--shot", required=True, type=int, choices=(5, 10, 20))
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=REPOSITORY_ROOT / "splits",
        help="Directory containing the version-controlled split manifests.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPOSITORY_ROOT / "results",
        help="Root directory for experiment results.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        required=True,
        help="Path to the official MedSAM ViT-B checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cuda:0 or cpu.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Permit nondeterministic kernels. Deterministic execution is the default.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size-gpps", type=int, default=2)
    parser.add_argument("--batch-size-lprs", type=int, default=1)
    parser.add_argument("--epochs-gpps", type=int, default=500)
    parser.add_argument("--epochs-lprs", type=int, default=500)
    parser.add_argument("--learning-rate-gpps", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate-lprs", type=float, default=3.0e-4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--base-patches", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=56)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sigma-scale", type=float, default=0.5)
    parser.add_argument("--sliding-window-batch-size", type=int, default=1)
    parser.add_argument("--halo-size", type=int, default=8)
    parser.add_argument(
        "--disable-halo-context",
        action="store_true",
        help="Disable the surrounding halo context used by the full model.",
    )
    parser.add_argument(
        "--use-mask-addition",
        action="store_true",
        help="Use the legacy RGB-plus-coarse-mask LPRS input.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("last", "train_loss"),
        default="last",
        help=(
            "Choose the final epoch by default, or the minimum annotated-training "
            "loss. Held-out samples are never used."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed gpps.pth and lprs.pth stages in the output directory.",
    )
    args = parser.parse_args(argv)

    for name in ("num_workers", "batch_size_gpps", "batch_size_lprs"):
        value = getattr(args, name)
        minimum = 0 if name == "num_workers" else 1
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}.")
    if not args.sam_checkpoint.is_file():
        parser.error(f"MedSAM checkpoint not found: {args.sam_checkpoint}")
    return args


def make_training_loader(
    dataset: KGLRDataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: str,
) -> DataLoader:
    """Build a deterministic shuffled loader over every selected annotation."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_data_worker,
        generator=make_data_generator(seed),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        args.result_root / args.dataset / f"{args.shot}_shot" / f"fold_{args.fold}" / "weights"
    )
    dataset = KGLRDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        split="train",
        shot=args.shot,
        fold=args.fold,
        augment=True,
        splits_root=args.splits_root,
    )
    if len(dataset) != args.shot:
        raise RuntimeError(f"Expected {args.shot} annotated samples, found {len(dataset)}.")

    gpps_loader = make_training_loader(
        dataset,
        batch_size=args.batch_size_gpps,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    lprs_loader = make_training_loader(
        dataset,
        batch_size=args.batch_size_lprs,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        device=args.device,
    )
    config = TrainingConfig(
        dataset=args.dataset,
        sam_checkpoint=str(args.sam_checkpoint.resolve()),
        output_dir=str(output_dir.resolve()),
        device=args.device,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        epochs_gpps=args.epochs_gpps,
        learning_rate_gpps=args.learning_rate_gpps,
        epochs_lprs=args.epochs_lprs,
        learning_rate_lprs=args.learning_rate_lprs,
        accumulation_steps=args.accumulation_steps,
        base_patches=args.base_patches,
        patch_size=args.patch_size,
        overlap=args.overlap,
        sigma_scale=args.sigma_scale,
        sliding_window_batch_size=args.sliding_window_batch_size,
        use_halo_context=not args.disable_halo_context,
        halo_size=args.halo_size,
        use_mask_addition=args.use_mask_addition,
        checkpoint_selection=args.checkpoint_selection,
        resume=args.resume,
    )
    gpps_path, lprs_path = train_two_stage(gpps_loader, config, lprs_loader=lprs_loader)
    print(f"GPPS checkpoint: {gpps_path}")
    print(f"LPRS checkpoint: {lprs_path}")
    print("Training finished without evaluating or selecting checkpoints on " "held-out data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

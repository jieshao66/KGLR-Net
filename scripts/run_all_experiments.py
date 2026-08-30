#!/usr/bin/env python3
"""Run the complete fixed-shot KGLR-Net training matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALL_DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")
ALL_SHOTS = (5, 10, 20)
ALL_FOLDS = (1, 2, 3, 4, 5)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train four datasets at 5, 10, and 20 shots over five fixed folds.")
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, default=REPOSITORY_ROOT / "splits")
    parser.add_argument("--result-root", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, choices=ALL_SHOTS, default=ALL_SHOTS)
    parser.add_argument("--folds", nargs="+", type=int, choices=ALL_FOLDS, default=ALL_FOLDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs-gpps", type=int, default=500)
    parser.add_argument("--epochs-lprs", type=int, default=500)
    parser.add_argument("--batch-size-gpps", type=int, default=2)
    parser.add_argument("--batch-size-lprs", type=int, default=1)
    parser.add_argument("--base-patches", type=int, default=3)
    parser.add_argument(
        "--disable-halo-context",
        action="store_true",
        help="Disable the surrounding halo context used by the full model.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("last", "train_loss"),
        default="last",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs and resume partially completed two-stage runs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    return parser.parse_args(argv)


def experiment_grid(
    datasets: Iterable[str], shots: Iterable[int], folds: Iterable[int]
) -> Iterable[tuple[str, int, int]]:
    for dataset in datasets:
        for shot in shots:
            for fold in folds:
                yield dataset, shot, fold


def build_command(args: argparse.Namespace, dataset: str, shot: int, fold: int) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "train.py"),
        "--data-root",
        str(args.data_root),
        "--dataset",
        dataset,
        "--shot",
        str(shot),
        "--fold",
        str(fold),
        "--splits-root",
        str(args.splits_root),
        "--result-root",
        str(args.result_root),
        "--sam-checkpoint",
        str(args.sam_checkpoint),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--epochs-gpps",
        str(args.epochs_gpps),
        "--epochs-lprs",
        str(args.epochs_lprs),
        "--batch-size-gpps",
        str(args.batch_size_gpps),
        "--batch-size-lprs",
        str(args.batch_size_lprs),
        "--base-patches",
        str(args.base_patches),
        "--checkpoint-selection",
        args.checkpoint_selection,
    ]
    if args.resume:
        command.append("--resume")
    if args.disable_halo_context:
        command.append("--disable-halo-context")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = list(experiment_grid(args.datasets, args.shots, args.folds))
    print(f"Scheduled experiments: {len(experiments)}")

    completed = 0
    skipped = 0
    for index, (dataset, shot, fold) in enumerate(experiments, start=1):
        weights_dir = args.result_root / dataset / f"{shot}_shot" / f"fold_{fold}" / "weights"
        finished = all((weights_dir / name).is_file() for name in ("gpps.pth", "lprs.pth"))
        label = f"{dataset} | {shot}-shot | fold {fold}"
        if args.resume and finished and not args.dry_run:
            print(f"[{index}/{len(experiments)}] Skip completed: {label}")
            skipped += 1
            continue

        command = build_command(args, dataset, shot, fold)
        print(f"[{index}/{len(experiments)}] {label}")
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            completed += 1

    if args.dry_run:
        print("Dry run complete; no training commands were executed.")
    else:
        print(f"Completed this invocation: {completed}; skipped: {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

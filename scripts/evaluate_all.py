"""Evaluate fixed checkpoints for the complete in-domain experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--shots", nargs="+", type=int, choices=(5, 10, 20), default=[5, 10, 20])
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-halo-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = len(args.datasets) * len(args.shots) * len(args.folds)
    index = 0
    for dataset in args.datasets:
        for shot in args.shots:
            for fold in args.folds:
                index += 1
                experiment = args.result_root / dataset / f"{shot}_shot" / f"fold_{fold}"
                weights = experiment / "weights"
                output = experiment / "metrics"
                summary = output / "summary.json"
                if args.skip_existing and summary.is_file():
                    print(f"[{index}/{total}] Skip existing: {summary}")
                    continue
                command = [
                    sys.executable,
                    str(REPOSITORY_ROOT / "evaluate.py"),
                    "--data-root",
                    str(args.data_root),
                    "--dataset",
                    dataset,
                    "--sam-checkpoint",
                    str(args.sam_checkpoint),
                    "--gpps-checkpoint",
                    str(weights / "gpps.pth"),
                    "--lprs-checkpoint",
                    str(weights / "lprs.pth"),
                    "--output-dir",
                    str(output),
                    "--device",
                    args.device,
                    "--num-workers",
                    str(args.num_workers),
                ]
                if args.disable_halo_context:
                    command.append("--disable-halo-context")
                print(f"[{index}/{total}] {dataset} | {shot}-shot | fold {fold}")
                print(subprocess.list2cmdline(command))
                if not args.dry_run:
                    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()

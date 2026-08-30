#!/usr/bin/env python
"""Generate the fixed-shot labeled splits used by KGLR-Net."""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")
SHOTS = (5, 10, 20)
FOLDS = (1, 2, 3, 4, 5)
BASE_SEED = 42
EXPECTED_COUNTS: Dict[str, Tuple[int, int]] = {
    "ISIC2018": (2594, 1000),
    "HAM10000": (8012, 2003),
    "Kvasir-SEG": (800, 100),
    "CVC-ClinicDB": (550, 62),
}


def read_manifest(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed partition manifest: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise RuntimeError(f"Manifest is empty: {path}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"Manifest contains duplicate entries: {path}")
    invalid = [name for name in names if Path(name).name != name or not name.endswith(".npz")]
    if invalid:
        raise ValueError(f"Manifest contains invalid .npz basenames: {invalid[:5]}")
    return names


def disk_names(directory: Path) -> Set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing preprocessed directory: {directory}")
    names = {path.name for path in directory.glob("*.npz")}
    if not names:
        raise RuntimeError(f"No .npz files found in {directory}")
    return names


def validated_partition(
    data_root: Path, splits_root: Path, dataset: str
) -> Tuple[List[str], Set[str]]:
    train_on_disk = disk_names(data_root / dataset / "train" / "npz_data")
    test_on_disk = disk_names(data_root / dataset / "test" / "npz_data")
    train_manifest = set(read_manifest(splits_root / dataset / "train_pool.txt"))
    test_manifest = set(read_manifest(splits_root / dataset / "test.txt"))

    expected_train, expected_test = EXPECTED_COUNTS[dataset]
    if len(train_manifest) != expected_train or len(test_manifest) != expected_test:
        raise RuntimeError(
            f"{dataset} manifest count mismatch: train={len(train_manifest)} "
            f"(expected {expected_train}), test={len(test_manifest)} "
            f"(expected {expected_test})"
        )
    if train_manifest & test_manifest:
        raise RuntimeError(f"{dataset} fixed train and test manifests overlap")
    if test_on_disk != test_manifest:
        raise RuntimeError(
            f"{dataset} test directory does not match test.txt: "
            f"missing={len(test_manifest - test_on_disk)}, "
            f"unexpected={len(test_on_disk - test_manifest)}"
        )

    disk_overlap = train_on_disk & test_on_disk
    if disk_overlap:
        raise RuntimeError(
            f"{dataset} train and test directories overlap by {len(disk_overlap)} samples"
        )
    if train_on_disk != train_manifest:
        raise RuntimeError(
            f"{dataset} train directory does not match train_pool.txt: "
            f"missing={len(train_manifest - train_on_disk)}, "
            f"unexpected={len(train_on_disk - train_manifest)}"
        )
    return sorted(train_on_disk), test_manifest


def write_split(path: Path, names: Sequence[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Split already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(names) + "\n")


def generate_dataset_splits(
    data_root: Path,
    splits_root: Path,
    dataset: str,
    overwrite: bool = False,
) -> None:
    candidates, test_names = validated_partition(data_root, splits_root, dataset)
    for shot in SHOTS:
        for fold in FOLDS:
            seed = BASE_SEED + 1000 * shot + fold
            shuffled = candidates.copy()
            random.Random(seed).shuffle(shuffled)
            labeled = shuffled[:shot]
            if len(labeled) != shot or len(set(labeled)) != shot:
                raise RuntimeError(
                    f"Internal split-generation failure for {dataset}/{shot}/fold_{fold}"
                )
            overlap = set(labeled) & test_names
            if overlap:
                raise RuntimeError(
                    f"Refusing to write a split with training/test overlap: {dataset}, "
                    f"{shot}-shot, fold {fold}"
                )
            path = splits_root / dataset / f"{shot}_shot" / f"fold_{fold}_labeled.txt"
            write_split(path, labeled, overwrite)
    print(
        f"Generated {len(SHOTS) * len(FOLDS)} fixed labeled splits for "
        f"{dataset} from {len(candidates)} training candidates"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 5-, 10-, and 20-shot labeled splits for five independent "
            "resamplings. The seed is fixed to 42 + 1000*shot + fold."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing <dataset>/{train,test}/npz_data.",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "splits",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        generate_dataset_splits(
            data_root=args.data_root,
            splits_root=args.splits_root,
            dataset=dataset,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

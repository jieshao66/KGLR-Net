#!/usr/bin/env python
"""Validate the fixed data partitions and every fixed-shot split."""

import argparse
from pathlib import Path
import numpy as np

from generate_splits import (
    DATASETS,
    EXPECTED_COUNTS,
    FOLDS,
    SHOTS,
    read_manifest,
    validated_partition,
)


def validate_npz(path: Path) -> None:
    required = {"image", "mask", "w_fg", "w_bg"}
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        image = np.asarray(payload["image"])
        mask = np.asarray(payload["mask"])
        w_fg = np.asarray(payload["w_fg"])
        w_bg = np.asarray(payload["w_bg"])
    if image.shape != (224, 224, 3):
        raise ValueError(f"{path}: image shape is {image.shape}, expected (224, 224, 3)")
    for key, value in (("mask", mask), ("w_fg", w_fg), ("w_bg", w_bg)):
        if value.shape != (224, 224):
            raise ValueError(f"{path}: {key} shape is {value.shape}, expected (224, 224)")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains NaN or infinity")
    if not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError(f"{path}: mask is not binary")
    if (w_fg < 0).any() or (w_fg > 1).any() or (w_bg < 0).any() or (w_bg > 1).any():
        raise ValueError(f"{path}: spatial weights must lie in [0, 1]")


def check_dataset(
    data_root: Path,
    splits_root: Path,
    dataset: str,
    inspect_arrays: bool,
) -> None:
    train_pool, test_names = validated_partition(data_root, splits_root, dataset)
    train_set = set(train_pool)
    train_dir = data_root / dataset / "train" / "npz_data"
    test_dir = data_root / dataset / "test" / "npz_data"

    split_files = 0
    for shot in SHOTS:
        for fold in FOLDS:
            path = splits_root / dataset / f"{shot}_shot" / f"fold_{fold}_labeled.txt"
            names = read_manifest(path)
            if len(names) != shot:
                raise RuntimeError(f"{path}: expected {shot} entries, found {len(names)}")
            selected = set(names)
            if not selected.issubset(train_set):
                raise RuntimeError(
                    f"{path}: {len(selected - train_set)} entries are outside train_pool.txt"
                )
            if selected & test_names:
                raise RuntimeError(f"{path}: labeled split overlaps the held-out test set")
            missing = [name for name in names if not (train_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"{path}: listed training files are missing: {missing[:5]}")
            split_files += 1

    if split_files != len(SHOTS) * len(FOLDS):
        raise RuntimeError(f"{dataset}: expected 15 split files, checked {split_files}")

    if inspect_arrays:
        for index, name in enumerate(train_pool, start=1):
            validate_npz(train_dir / name)
            if index % 500 == 0:
                print(f"  inspected {index}/{len(train_pool)} training arrays")
        for index, name in enumerate(sorted(test_names), start=1):
            validate_npz(test_dir / name)
            if index % 500 == 0:
                print(f"  inspected {index}/{len(test_names)} test arrays")

    expected_train, expected_test = EXPECTED_COUNTS[dataset]
    print(
        f"PASS {dataset}: train_pool={expected_train}, test={expected_test}, "
        f"fixed-shot files={split_files}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check fixed partition manifests, all 60 labeled splits, and data files."
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
    parser.add_argument(
        "--inspect-arrays",
        action="store_true",
        help="Load every fixed train/test .npz file and validate its keys, shapes, and ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        check_dataset(
            data_root=args.data_root,
            splits_root=args.splits_root,
            dataset=dataset,
            inspect_arrays=args.inspect_arrays,
        )
    expected_files = len(args.datasets) * len(SHOTS) * len(FOLDS)
    print(f"All checks passed ({expected_files} fixed-shot split files).")


if __name__ == "__main__":
    main()

"""Dataset utilities for the limited-annotation KGLR-Net experiments.

Only fully supervised training samples and held-out test samples are exposed by
this module.  It deliberately contains no unlabeled-data, pseudo-label, or
student-training code.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_DATASETS = (
    "ISIC2018",
    "HAM10000",
    "Kvasir-SEG",
    "CVC-ClinicDB",
)
SUPPORTED_SHOTS = (5, 10, 20)
SUPPORTED_FOLDS = (1, 2, 3, 4, 5)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

PathLike = Union[str, Path]


def _read_name_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample list: {path}")

    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise RuntimeError(f"Sample list is empty: {path}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"Sample list contains duplicate entries: {path}")

    for name in names:
        candidate = Path(name)
        if candidate.name != name or candidate.suffix.lower() != ".npz":
            raise ValueError(f"Each entry must be an .npz basename, but found {name!r} in {path}")
    return names


class KGLRDataset(Dataset):
    """Load preprocessed samples for KGLR-Net.

    Parameters
    ----------
    data_root:
        Directory containing ``<dataset>/<train|test>/npz_data``.
    dataset:
        One of the four datasets evaluated in the paper.
    split:
        ``"train"`` loads one labeled fixed-shot list. ``"test"`` loads the
        complete held-out test partition.
    shot:
        Labeled training budget. Required for ``split="train"`` and restricted
        to 5, 10, or 20.
    fold:
        Independent resampling index in 1--5.
    augment:
        Apply synchronized random horizontal and vertical flips. Augmentation
        is permitted only for the training split.
    splits_root:
        Root of the version-controlled split manifests. By default, this is the
        repository's ``splits`` directory.

    Returns
    -------
    dict
        ``image`` is an ImageNet-normalized float tensor of shape ``[3,H,W]``;
        ``mask``, ``w_fg``, and ``w_bg`` are float tensors of shape ``[1,H,W]``;
        ``name``, ``dataset``, and ``split`` provide sample metadata.
    """

    def __init__(
        self,
        data_root: PathLike,
        dataset: str,
        split: str = "train",
        shot: Optional[int] = None,
        fold: int = 1,
        augment: bool = False,
        splits_root: Optional[PathLike] = None,
    ) -> None:
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset {dataset!r}; choose from {SUPPORTED_DATASETS}")
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")
        if augment and split != "train":
            raise ValueError("Random augmentation is available only for training")

        self.data_root = Path(data_root).expanduser()
        self.dataset = dataset
        self.split = split
        self.shot = shot
        self.fold = fold
        self.augment = augment
        self.splits_root = (
            Path(splits_root).expanduser()
            if splits_root is not None
            else Path(__file__).resolve().parents[1] / "splits"
        )
        self.npz_dir = self.data_root / dataset / split / "npz_data"

        if not self.npz_dir.is_dir():
            raise FileNotFoundError(f"Missing preprocessed data directory: {self.npz_dir}")

        if split == "train":
            if shot not in SUPPORTED_SHOTS:
                raise ValueError(
                    f"shot is required for training and must be one of {SUPPORTED_SHOTS}"
                )
            if fold not in SUPPORTED_FOLDS:
                raise ValueError(f"fold must be one of {SUPPORTED_FOLDS}")
            split_file = self.splits_root / dataset / f"{shot}_shot" / f"fold_{fold}_labeled.txt"
            self.file_names = _read_name_list(split_file)
            if len(self.file_names) != shot:
                raise RuntimeError(
                    f"Expected {shot} names in {split_file}, found {len(self.file_names)}"
                )
            self._assert_training_partition(self.file_names)
        else:
            if shot is not None:
                raise ValueError("shot must be omitted for the test split")
            self.file_names = sorted(path.name for path in self.npz_dir.glob("*.npz"))
            if not self.file_names:
                raise RuntimeError(f"No .npz samples found in {self.npz_dir}")
            self._assert_test_manifest_matches()

        missing = [name for name in self.file_names if not (self.npz_dir / name).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} listed samples are missing from {self.npz_dir}: {preview}"
            )

    def _assert_training_partition(self, training_names: List[str]) -> None:
        test_dir = self.data_root / self.dataset / "test" / "npz_data"
        if not test_dir.is_dir():
            raise FileNotFoundError(
                "A held-out test directory is required for partition validation: " f"{test_dir}"
            )
        test_names = {path.name for path in test_dir.glob("*.npz")}
        test_manifest = set(_read_name_list(self.splits_root / self.dataset / "test.txt"))
        if test_names != test_manifest:
            raise RuntimeError(
                f"Test data for {self.dataset} do not match the fixed test.txt manifest"
            )
        train_pool = set(_read_name_list(self.splits_root / self.dataset / "train_pool.txt"))
        outside_pool = sorted(set(training_names) - train_pool)
        if outside_pool:
            preview = ", ".join(outside_pool[:5])
            raise RuntimeError(
                f"Training split contains {len(outside_pool)} sample(s) outside "
                f"train_pool.txt, including {preview}"
            )
        overlap = sorted(set(training_names) & test_names)
        if overlap:
            preview = ", ".join(overlap[:5])
            raise RuntimeError(
                "Training split overlaps the held-out test partition: "
                f"{len(overlap)} sample(s), including {preview}"
            )

    def _assert_test_manifest_matches(self) -> None:
        manifest = self.splits_root / self.dataset / "test.txt"
        if not manifest.is_file():
            return
        expected = set(_read_name_list(manifest))
        actual = set(self.file_names)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"Test data do not match {manifest}: "
                f"missing={len(missing)}, unexpected={len(extra)}"
            )

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> Dict[str, object]:
        name = self.file_names[index]
        path = self.npz_dir / name
        required = {"image", "mask", "w_fg", "w_bg"}
        with np.load(path, allow_pickle=False) as payload:
            missing = required - set(payload.files)
            if missing:
                raise KeyError(f"{path} is missing keys: {sorted(missing)}")
            image = np.asarray(payload["image"])
            mask = np.asarray(payload["mask"])
            w_fg = np.asarray(payload["w_fg"], dtype=np.float32)
            w_bg = np.asarray(payload["w_bg"], dtype=np.float32)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 image in {path}, found {image.shape}")
        height, width = image.shape[:2]
        for field_name, field in (("mask", mask), ("w_fg", w_fg), ("w_bg", w_bg)):
            if field.shape != (height, width):
                raise ValueError(
                    f"{field_name} in {path} has shape {field.shape}; "
                    f"expected {(height, width)}"
                )
            if not np.isfinite(field).all():
                raise ValueError(f"{field_name} in {path} contains NaN or infinity")

        mask = (mask > 0.5).astype(np.float32)
        if (w_fg < 0).any() or (w_fg > 1).any() or (w_bg < 0).any() or (w_bg > 1).any():
            raise ValueError(f"Spatial weights in {path} must lie in [0, 1]")

        if self.augment:
            if bool(torch.rand(()) < 0.5):
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
                w_fg = np.flip(w_fg, axis=1)
                w_bg = np.flip(w_bg, axis=1)
            if bool(torch.rand(()) < 0.5):
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)
                w_fg = np.flip(w_fg, axis=0)
                w_bg = np.flip(w_bg, axis=0)

        image = np.ascontiguousarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image /= 255.0
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(f"Image values in {path} must lie in [0, 255] or [0, 1]")
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
            "w_fg": torch.from_numpy(np.ascontiguousarray(w_fg[None])).float(),
            "w_bg": torch.from_numpy(np.ascontiguousarray(w_bg[None])).float(),
            "name": Path(name).stem,
            "dataset": self.dataset,
            "split": self.split,
        }

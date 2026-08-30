#!/usr/bin/env python
"""Convert one raw image/mask partition into safe KGLR-Net .npz samples."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_SIZE = (224, 224)


def calculate_spatial_weights(
    mask: np.ndarray,
    sigma: float = 15.0,
    alpha_fg: float = 0.1,
    alpha_bg: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the foreground- and background-oriented spatial weight maps."""
    mask = (np.asarray(mask) > 0.5).astype(np.uint8)
    if mask.max() == 0:
        return (
            np.full(mask.shape, alpha_fg, dtype=np.float32),
            np.ones(mask.shape, dtype=np.float32),
        )

    distance = distance_transform_edt(1 - mask)
    outside = mask == 0

    w_fg = np.ones_like(distance, dtype=np.float32)
    w_fg[outside] = alpha_fg + (1.0 - alpha_fg) * np.exp(
        -(distance[outside] ** 2) / (2.0 * sigma**2)
    )

    w_bg = np.full_like(distance, alpha_bg, dtype=np.float32)
    max_distance = float(distance.max())
    if max_distance > 0:
        normalized_distance = distance / max_distance
        w_bg[outside] = alpha_bg + (1.0 - alpha_bg) * normalized_distance[outside]
    return w_fg, w_bg


def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No supported image files found in {directory}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"Duplicate image stems found in {directory}")
    return paths


def _mask_index(mask_dir: Path) -> Dict[str, Path]:
    masks = _image_files(mask_dir)
    index: Dict[str, Path] = {}
    for path in masks:
        key = path.stem
        if key in index:
            raise RuntimeError(f"Duplicate mask stem {key!r} in {mask_dir}")
        index[key] = path
    return index


def _find_mask(image_stem: str, masks: Dict[str, Path]) -> Path:
    candidates = [
        key
        for key in (
            image_stem,
            f"{image_stem}_mask",
            f"{image_stem}_segmentation",
        )
        if key in masks
    ]
    if not candidates:
        raise FileNotFoundError(f"No mask matches image stem {image_stem!r}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple masks match image stem {image_stem!r}: {candidates}. "
            "The four supported datasets require one binary mask per image."
        )
    return masks[candidates[0]]


def _read_with_opencv(path: Path, flags: int) -> np.ndarray:
    """Read an image through memory so non-ASCII Windows paths are supported."""
    encoded = np.fromfile(path, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, flags)
    if decoded is None:
        raise RuntimeError(f"OpenCV could not read file: {path}")
    return decoded


def preprocess_partition(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    """Preprocess one already-defined train or test partition."""
    image_paths = _image_files(image_dir)
    masks = _mask_index(mask_dir)
    npz_dir = output_dir / "npz_data"
    npz_dir.mkdir(parents=True, exist_ok=True)

    existing = list(npz_dir.glob("*.npz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{npz_dir} already contains {len(existing)} .npz files. "
            "Use --overwrite only after confirming the target directory."
        )

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = _find_mask(image_path.stem, masks)
        image_bgr = _read_with_opencv(image_path, cv2.IMREAD_COLOR)
        mask_raw = _read_with_opencv(mask_path, cv2.IMREAD_GRAYSCALE)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_raw, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        threshold = 127 if int(mask_resized.max()) > 127 else 0
        mask = (mask_resized > threshold).astype(np.uint8)
        w_fg, w_bg = calculate_spatial_weights(mask)

        np.savez_compressed(
            npz_dir / f"{image_path.stem}.npz",
            image=image.astype(np.uint8, copy=False),
            mask=mask,
            w_fg=w_fg,
            w_bg=w_bg,
        )
        if index % 100 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)} samples")
    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize one fixed dataset partition to 224x224 and save image, mask, "
            "foreground weight, and background weight arrays in a non-pickle .npz file."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root containing <dataset>/<split>/npz_data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of existing .npz files in the selected partition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.dataset / args.split
    count = preprocess_partition(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    print(
        f"Finished {args.dataset}/{args.split}: {count} samples written to "
        f"{output_dir / 'npz_data'}"
    )


if __name__ == "__main__":
    main()

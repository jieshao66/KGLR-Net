"""Run KGLR-Net on one image or a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as vision_functional
from tqdm import tqdm

from kglr_net.pipeline import KGLRPipeline
from kglr_net.runtime import resolve_device


DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
        help=(
            "Selects the modality-specific prior branch; use a dataset of the "
            "same modality as the input."
        ),
    )
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--gpps-checkpoint", type=Path, required=True)
    parser.add_argument("--lprs-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=56)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sigma-scale", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument(
        "--disable-halo-context",
        action="store_true",
        help="Disable the halo context used by the full paper configuration.",
    )
    parser.add_argument("--halo-size", type=int, default=8)
    parser.add_argument("--use-mask-addition", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


def discover_images(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No supported image files were found in {path}.")
    return images


def prepare_image(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    tensor = vision_functional.to_tensor(resized)
    return vision_functional.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)


def save_mask(
    probability: np.ndarray, path: Path, original_size: tuple[int, int], threshold: float
) -> None:
    binary = Image.fromarray((probability > threshold).astype(np.uint8) * 255, mode="L")
    binary.resize(original_size, Image.Resampling.NEAREST).save(path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    pipeline = KGLRPipeline(
        dataset=args.dataset,
        sam_checkpoint=args.sam_checkpoint,
        gpps_checkpoint=args.gpps_checkpoint,
        lprs_checkpoint=args.lprs_checkpoint,
        patch_size=args.patch_size,
        overlap=args.overlap,
        sigma_scale=args.sigma_scale,
        sw_batch_size=args.sw_batch_size,
        use_halo_context=not args.disable_halo_context,
        halo_size=args.halo_size,
        use_mask_addition=args.use_mask_addition,
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(discover_images(args.input), desc="Inference"):
        with Image.open(image_path) as image:
            original_size = image.size
            tensor = prepare_image(image, args.image_size).to(device)
        predictions = pipeline(tensor)
        global_probability = predictions["gpps"].squeeze().cpu().numpy()
        refined_probability = predictions["kglr"].squeeze().cpu().numpy()
        save_mask(
            global_probability,
            args.output_dir / f"{image_path.stem}_gpps.png",
            original_size,
            args.threshold,
        )
        save_mask(
            refined_probability,
            args.output_dir / f"{image_path.stem}_kglr.png",
            original_size,
            args.threshold,
        )
        if args.save_probabilities:
            np.save(args.output_dir / f"{image_path.stem}_gpps.npy", global_probability)
            np.save(args.output_dir / f"{image_path.stem}_kglr.npy", refined_probability)


if __name__ == "__main__":
    main()

"""Evaluate source-trained KGLR-Net weights on a target dataset without adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kglr_net.data import KGLRDataset
from kglr_net.metrics import compute_segmentation_metrics, metrics_to_dict, summarize_rows
from kglr_net.pipeline import KGLRPipeline
from kglr_net.runtime import resolve_device, write_csv, write_json


DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")
MODALITY_PAIRS = {
    "ISIC2018": "HAM10000",
    "HAM10000": "ISIC2018",
    "Kvasir-SEG": "CVC-ClinicDB",
    "CVC-ClinicDB": "Kvasir-SEG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--source-dataset", choices=DATASETS, required=True)
    parser.add_argument("--target-dataset", choices=DATASETS, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--gpps-checkpoint", type=Path, required=True)
    parser.add_argument("--lprs-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
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
    parser.add_argument(
        "--allow-cross-modality",
        action="store_true",
        help=(
            "Allow an evaluation pair outside the within-modality directions "
            "reported in the paper."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_target = MODALITY_PAIRS[args.source_dataset]
    if args.target_dataset != expected_target and not args.allow_cross_modality:
        raise ValueError(
            f"The paper evaluates {args.source_dataset} -> {expected_target}. "
            "Use --allow-cross-modality only for an explicitly exploratory run."
        )

    device = resolve_device(args.device)
    target_set = KGLRDataset(
        data_root=args.data_root,
        dataset=args.target_dataset,
        split="test",
        augment=False,
    )
    loader = DataLoader(
        target_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    pipeline = KGLRPipeline(
        dataset=args.target_dataset,
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

    gpps_rows: List[Dict[str, float | str]] = []
    kglr_rows: List[Dict[str, float | str]] = []
    direction = f"{args.source_dataset}->{args.target_dataset}"
    for batch in tqdm(loader, desc=direction):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].squeeze().cpu().numpy()
        name = batch["name"][0]
        predictions = pipeline(image)
        for key, rows in (("gpps", gpps_rows), ("kglr", kglr_rows)):
            prediction = predictions[key].squeeze().cpu().numpy()
            rows.append(
                {
                    "file": name,
                    **metrics_to_dict(
                        compute_segmentation_metrics(prediction, target, threshold=args.threshold)
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gpps_cross_per_image.csv", gpps_rows)
    write_csv(args.output_dir / "kglr_net_cross_per_image.csv", kglr_rows)
    write_json(
        args.output_dir / "cross_summary.json",
        {
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "num_images": len(target_set),
            "threshold": args.threshold,
            "gpps": summarize_rows(gpps_rows),
            "kglr_net": summarize_rows(kglr_rows),
        },
    )


if __name__ == "__main__":
    main()

"""Training utilities for the two-stage KGLR-Net pipeline.

This module trains GPPS first and then LPRS using only the annotated samples
provided by the training data loader. It deliberately contains no evaluation,
pseudo-label generation, unlabeled-data path, or student-model path.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import (
    GlobalPriorPromptingSegmenter,
    LocalPriorRefinementSegmenter,
    SignalPriorExtractor,
    get_signal_domain,
)


@dataclass
class TrainingConfig:
    """Configuration shared by GPPS and LPRS training."""

    dataset: str
    sam_checkpoint: str
    output_dir: str
    device: str = "cuda:0"
    seed: int = 42
    deterministic: bool = True
    model_type: str = "vit_b"
    lora_rank: int = 4

    epochs_gpps: int = 500
    learning_rate_gpps: float = 1.0e-4
    epochs_lprs: int = 500
    learning_rate_lprs: float = 3.0e-4
    accumulation_steps: int = 4

    base_patches: int = 3
    patch_size: int = 56
    overlap: float = 0.5
    sigma_scale: float = 0.5
    sliding_window_batch_size: int = 1
    use_halo_context: bool = True
    halo_size: int = 8
    use_mask_addition: bool = False

    lambda_heads_lprs: float = 1.0
    lambda_final_lprs: float = 1.0
    lambda_base_lprs: float = 0.5
    lambda_route_balance_lprs: float = 1.0e-2
    lambda_kl_lprs: float = 1.0e-3

    checkpoint_selection: str = "last"
    resume: bool = False

    def validate(self) -> None:
        if self.dataset not in {
            "ISIC2018",
            "HAM10000",
            "Kvasir-SEG",
            "CVC-ClinicDB",
        }:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.epochs_gpps < 1 or self.epochs_lprs < 1:
            raise ValueError("Both training stages require at least one epoch.")
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive.")
        if self.base_patches < 1:
            raise ValueError("base_patches must be positive.")
        if self.patch_size < 1 or self.halo_size < 0:
            raise ValueError("Invalid patch or halo size.")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1).")
        if self.sigma_scale <= 0.0:
            raise ValueError("sigma_scale must be positive.")
        if self.checkpoint_selection not in {"last", "train_loss"}:
            raise ValueError("checkpoint_selection must be either 'last' or 'train_loss'.")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic kernels."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_worker(worker_id: int) -> None:
    """Seed a data-loader worker from the seed assigned by PyTorch."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_generator(seed: int) -> torch.Generator:
    """Create the generator used for deterministic data shuffling."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _safe_probability(tensor: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.5, posinf=1.0 - eps, neginf=eps)
    return tensor.clamp(min=eps, max=1.0 - eps)


def _safe_target(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=1.0, neginf=0.0)
    return tensor.clamp(min=0.0, max=1.0)


def dice_loss(
    prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1.0e-5
) -> torch.Tensor:
    prediction = _safe_probability(prediction)
    target = _safe_target(target)
    intersection = (prediction * target).sum(dim=(2, 3))
    denominator = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def route_balance_loss(
    route_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize early collapse of the five-expert routing distribution."""

    if route_weights.ndim != 5 or route_weights.shape[1] != 5:
        raise ValueError("route_weights must have shape [batch, 5, 1, height, width].")
    usage = route_weights.mean(dim=(0, 2, 3, 4))
    uniform = torch.full_like(usage, 1.0 / route_weights.shape[1])
    return F.mse_loss(usage, uniform), usage.detach()


def normalized_kl_loss(
    means: Sequence[torch.Tensor], log_variances: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Average normalized KL regularization over the probabilistic features."""

    if not means or len(means) != len(log_variances):
        raise ValueError("The mean and log-variance feature lists must match.")
    terms: List[torch.Tensor] = []
    for mean, log_variance in zip(means, log_variances):
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = mean.clamp(-10.0, 10.0)
        log_variance = torch.nan_to_num(log_variance, nan=0.0, posinf=10.0, neginf=-10.0).clamp(
            -10.0, 10.0
        )
        terms.append(-0.5 * torch.mean(1.0 + log_variance - mean.pow(2) - log_variance.exp()))
    return torch.stack(terms).mean()


def center_crop_last_two_dims(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Crop a centered square from the final two dimensions."""

    height, width = tensor.shape[-2:]
    if crop_size > height or crop_size > width:
        raise ValueError(f"crop_size={crop_size} exceeds spatial shape {(height, width)}.")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return tensor[..., top : top + crop_size, left : left + crop_size]


def _patch_dice(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    pred_binary = (prediction.detach() > 0.5).float()
    target_binary = (target.detach() > 0.5).float()
    true_positive = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    false_positive = (pred_binary * (1.0 - target_binary)).sum(dim=(1, 2, 3))
    false_negative = ((1.0 - pred_binary) * target_binary).sum(dim=(1, 2, 3))
    dice = (2.0 * true_positive + eps) / (
        2.0 * true_positive + false_positive + false_negative + eps
    )
    return dice.mean()


def boundary_proxy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    smooth: float = 1.0e-5,
) -> torch.Tensor:
    """Compute a training-only boundary proxy used for monitoring."""

    prediction = _safe_probability(prediction)
    target = _safe_target(target)
    dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded > 0.05).float()
    if boundary.sum().item() < 1.0:
        return criterion(prediction, target) + dice_loss(prediction, target)
    bce = F.binary_cross_entropy(
        prediction, target, weight=boundary, reduction="sum"
    ) / boundary.sum().clamp_min(1.0)
    intersection = (prediction * target * boundary).sum(dim=(1, 2, 3))
    denominator = ((prediction + target) * boundary).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return bce + (1.0 - dice.mean())


def _accumulation_divisor(batch_index: int, number_of_batches: int, accumulation_steps: int) -> int:
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, number_of_batches - group_start)


def _validate_training_batch(batch: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    if "image" not in batch or "mask" not in batch:
        raise KeyError("A training batch must contain 'image' and 'mask'.")
    image = batch["image"]
    mask = batch["mask"]
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("The image and mask batch entries must be tensors.")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Images must have shape [batch, 3, height, width].")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Masks must have shape [batch, 1, height, width].")
    if image.shape[0] != mask.shape[0] or image.shape[-2:] != mask.shape[-2:]:
        raise ValueError("Image and mask batch dimensions must match.")
    return image, mask


def sample_high_value_patches(
    tensors: Sequence[torch.Tensor],
    patch_size: int = 56,
    base_patches: int = 3,
    use_halo_context: bool = True,
    halo_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample boundary, interior, GPPS-error, and safe-background patches.

    The inputs must be image, GPPS probability, texture priors, intensity
    priors, and ground-truth mask in that order. Ground truth is used only for
    patch selection during training.
    """

    if len(tensors) != 5:
        raise ValueError("Expected image, GPPS mask, two priors, and target mask.")
    batch_size, _, height, width = tensors[0].shape
    expected_channels = (3, 1, 9, 3, 1)
    for tensor, channels in zip(tensors, expected_channels):
        if tensor.ndim != 4 or tensor.shape[0] != batch_size:
            raise ValueError("All patch-sampling inputs must be aligned 4-D tensors.")
        if tensor.shape[1] != channels or tensor.shape[-2:] != (height, width):
            raise ValueError("Unexpected channel count or spatial shape.")
        if tensor.device != tensors[0].device:
            raise ValueError("All patch-sampling inputs must be on one device.")

    if patch_size > height or patch_size > width:
        raise ValueError("patch_size cannot exceed the image size.")
    effective_halo = halo_size if use_halo_context else 0
    context_size = patch_size + 2 * effective_halo
    if context_size % 8 != 0:
        raise ValueError("The LPRS context size must be divisible by eight.")
    if effective_halo >= height or effective_halo >= width:
        raise ValueError("halo_size must be smaller than the image dimensions.")

    if effective_halo:
        padded_inputs: Sequence[torch.Tensor] = [
            F.pad(
                tensor,
                (effective_halo, effective_halo, effective_halo, effective_halo),
                mode="reflect",
            )
            for tensor in tensors[:-1]
        ]
    else:
        padded_inputs = tensors[:-1]

    output_inputs: List[List[torch.Tensor]] = [[] for _ in padded_inputs]
    output_targets: List[torch.Tensor] = []
    gpps_probability = tensors[1]
    target_tensor = tensors[4]
    total_area = height * width

    def choose_center(primary: torch.Tensor, fallback: torch.Tensor) -> Tuple[int, int]:
        candidates = primary if primary.shape[0] else fallback
        if not candidates.shape[0]:
            return random.randint(0, height - 1), random.randint(0, width - 1)
        selected = candidates[random.randint(0, candidates.shape[0] - 1)]
        return int(selected[0].item()), int(selected[1].item())

    for image_index in range(batch_size):
        target = target_tensor[image_index, 0]
        coarse = gpps_probability[image_index, 0]
        target_binary = target > 0.5
        coarse_binary = coarse > 0.5
        target_4d = target[None, None]
        dilated = F.max_pool2d(target_4d, 3, stride=1, padding=1)[0, 0]
        eroded = -F.max_pool2d(-target_4d, 3, stride=1, padding=1)[0, 0]
        boundary = dilated - eroded > 0.05
        difficult = (coarse_binary != target_binary) | (torch.abs(coarse - 0.5) < 0.15)
        safe_background = (~target_binary) & (dilated < 0.5)

        lesion_indices = torch.nonzero(target_binary, as_tuple=False)
        boundary_indices = torch.nonzero(boundary, as_tuple=False)
        difficult_indices = torch.nonzero(difficult, as_tuple=False)
        background_indices = torch.nonzero(safe_background, as_tuple=False)

        lesion_area = lesion_indices.shape[0]
        if lesion_area == 0:
            number_of_patches = max(base_patches, 4)
        else:
            area_ratio = lesion_area / total_area
            if area_ratio < 0.1:
                number_of_patches = max(base_patches + 2, 6)
            elif area_ratio > 0.4:
                number_of_patches = base_patches * 2
            else:
                number_of_patches = base_patches

        jitter = max(1, patch_size // 8)
        for _ in range(number_of_patches):
            draw = random.random()
            if draw < 0.40:
                center_y, center_x = choose_center(boundary_indices, lesion_indices)
            elif draw < 0.65:
                center_y, center_x = choose_center(lesion_indices, boundary_indices)
            elif draw < 0.85:
                center_y, center_x = choose_center(difficult_indices, boundary_indices)
            else:
                center_y, center_x = choose_center(background_indices, difficult_indices)

            top = center_y - patch_size // 2 + random.randint(-jitter, jitter)
            left = center_x - patch_size // 2 + random.randint(-jitter, jitter)
            top = max(0, min(top, height - patch_size))
            left = max(0, min(left, width - patch_size))

            for tensor_index, tensor in enumerate(padded_inputs):
                output_inputs[tensor_index].append(
                    tensor[
                        image_index : image_index + 1,
                        :,
                        top : top + context_size,
                        left : left + context_size,
                    ]
                )
            output_targets.append(
                target_tensor[
                    image_index : image_index + 1,
                    :,
                    top : top + patch_size,
                    left : left + patch_size,
                ]
            )

    return (
        *(torch.cat(items, dim=0) for items in output_inputs),
        torch.cat(output_targets, dim=0),
    )


def _load_checkpoint(path: Path, map_location: str) -> Dict[str, torch.Tensor]:
    """Load a state dictionary with safe loading on supported PyTorch versions."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    state = dict(payload)
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def _save_state(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _selected_checkpoint(output_dir: Path, stage: str, selection: str) -> Path:
    suffix = "last" if selection == "last" else "best_train_loss"
    return output_dir / f"{stage}_{suffix}.pth"


def train_gpps(training_loader: DataLoader, config: TrainingConfig) -> Path:
    """Train GPPS using only annotated training samples."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = GlobalPriorPromptingSegmenter(
        model_type=config.model_type,
        checkpoint_path=config.sam_checkpoint,
        lora_rank=config.lora_rank,
        signal_domain=get_signal_domain(config.dataset),
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("GPPS has no trainable parameters.")
    optimizer = AdamW(trainable_parameters, lr=config.learning_rate_gpps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs_gpps, eta_min=1.0e-6
    )
    criterion = nn.BCELoss()
    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    last_path = output_dir / "gpps_last.pth"
    best_path = output_dir / "gpps_best_train_loss.pth"

    for epoch in range(1, config.epochs_gpps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        progress = tqdm(
            training_loader,
            desc=f"GPPS epoch {epoch}/{config.epochs_gpps}",
            leave=False,
        )
        for batch_index, batch in enumerate(progress):
            images, masks = _validate_training_batch(batch)
            images = images.to(device, non_blocking=True)
            masks = _safe_target(masks.to(device, non_blocking=True))
            predictions = _safe_probability(model(images))
            loss = criterion(predictions, masks) + dice_loss(predictions, masks)
            divisor = _accumulation_divisor(
                batch_index, len(training_loader), config.accumulation_steps
            )
            (loss / divisor).backward()
            final_batch = batch_index + 1 == len(training_loader)
            if (batch_index + 1) % config.accumulation_steps == 0 or final_batch:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().item())
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

        average_loss = total_loss / max(1, len(training_loader))
        scheduler.step()
        _save_state(model, last_path)
        if average_loss < best_loss:
            best_loss = average_loss
            _save_state(model, best_path)
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(f"GPPS epoch {epoch}: train_loss={average_loss:.6f}")

    selected = _selected_checkpoint(output_dir, "gpps", config.checkpoint_selection)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    shutil.copy2(selected, output_dir / "gpps.pth")
    _write_json(output_dir / "gpps_history.json", history)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir / "gpps.pth"


def train_lprs(
    training_loader: DataLoader,
    config: TrainingConfig,
    gpps_checkpoint: Path,
) -> Path:
    """Train LPRS using frozen GPPS predictions and annotated samples."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    domain = get_signal_domain(config.dataset)

    gpps = GlobalPriorPromptingSegmenter(
        model_type=config.model_type,
        checkpoint_path=config.sam_checkpoint,
        lora_rank=config.lora_rank,
        signal_domain=domain,
    ).to(device)
    gpps.load_state_dict(_load_checkpoint(gpps_checkpoint, config.device), strict=True)
    gpps.eval()
    for parameter in gpps.parameters():
        parameter.requires_grad = False

    extractor = SignalPriorExtractor(domain=domain).to(device)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad = False

    lprs = LocalPriorRefinementSegmenter(use_mask_addition=config.use_mask_addition).to(device)
    optimizer = AdamW(lprs.parameters(), lr=config.learning_rate_lprs)
    if config.epochs_lprs > 1:
        warmup_epochs = min(max(1, int(config.epochs_lprs * 0.05)), config.epochs_lprs - 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs_lprs - warmup_epochs),
            eta_min=1.0e-6,
        )
        scheduler: Optional[
            torch.optim.lr_scheduler.LRScheduler
        ] = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = None

    criterion = nn.BCELoss()
    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    last_path = output_dir / "lprs_last.pth"
    best_path = output_dir / "lprs_best_train_loss.pth"

    for epoch in range(1, config.epochs_lprs + 1):
        lprs.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "heads": 0.0,
            "base": 0.0,
            "final": 0.0,
            "boundary": 0.0,
            "route": 0.0,
            "kl": 0.0,
            "total": 0.0,
            "gpps_patch_dice": 0.0,
            "lprs_patch_dice": 0.0,
        }
        progress = tqdm(
            training_loader,
            desc=f"LPRS epoch {epoch}/{config.epochs_lprs}",
            leave=False,
        )
        for batch_index, batch in enumerate(progress):
            images, masks = _validate_training_batch(batch)
            images = images.to(device, non_blocking=True)
            masks = _safe_target(masks.to(device, non_blocking=True))

            with torch.no_grad():
                coarse_masks = _safe_probability(gpps(images))
                texture_priors, intensity_priors = extractor(images, teacher_a_prob=coarse_masks)

            (
                image_patch,
                coarse_patch,
                texture_patch,
                intensity_patch,
                target_patch,
            ) = sample_high_value_patches(
                [images, coarse_masks, texture_priors, intensity_priors, masks],
                patch_size=config.patch_size,
                base_patches=config.base_patches,
                use_halo_context=config.use_halo_context,
                halo_size=config.halo_size,
            )
            image_patch = torch.nan_to_num(image_patch)
            coarse_patch = _safe_probability(coarse_patch)
            texture_patch = torch.nan_to_num(texture_patch)
            intensity_patch = torch.nan_to_num(intensity_patch)
            target_patch = _safe_target(target_patch)

            output = lprs(
                image_patch,
                coarse_patch,
                texture_patch,
                intensity_patch,
                return_aux=True,
            )
            if not isinstance(output, tuple) or len(output) != 6:
                raise RuntimeError("LPRS must return six values when return_aux=True.")
            predictions, latent, final_mask, _, _, auxiliary = output
            means, log_variances = latent

            prediction_cores = [
                _safe_probability(center_crop_last_two_dims(item, config.patch_size))
                for item in predictions
            ]
            final_core = _safe_probability(center_crop_last_two_dims(final_mask, config.patch_size))
            base_core = _safe_probability(
                center_crop_last_two_dims(auxiliary["m_b_base"], config.patch_size)
            )
            route_core = center_crop_last_two_dims(auxiliary["route_weights"], config.patch_size)
            route_core = torch.nan_to_num(route_core, nan=0.0, posinf=1.0, neginf=0.0).clamp(
                0.0, 1.0
            )
            coarse_core = _safe_probability(
                center_crop_last_two_dims(coarse_patch, config.patch_size)
            )

            loss_heads = sum(
                criterion(item, target_patch) + dice_loss(item, target_patch)
                for item in prediction_cores
            ) / len(prediction_cores)
            loss_final = criterion(final_core, target_patch) + dice_loss(final_core, target_patch)
            loss_base = criterion(base_core, target_patch) + dice_loss(base_core, target_patch)
            loss_boundary = boundary_proxy_loss(final_core, target_patch, criterion)
            loss_route, _ = route_balance_loss(route_core)
            loss_kl = normalized_kl_loss(means, log_variances)
            loss = (
                config.lambda_heads_lprs * loss_heads
                + config.lambda_base_lprs * loss_base
                + config.lambda_final_lprs * loss_final
                + config.lambda_route_balance_lprs * loss_route
                + config.lambda_kl_lprs * loss_kl
            )

            divisor = _accumulation_divisor(
                batch_index, len(training_loader), config.accumulation_steps
            )
            (loss / divisor).backward()
            final_batch = batch_index + 1 == len(training_loader)
            if (batch_index + 1) % config.accumulation_steps == 0 or final_batch:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                gpps_patch_dice = _patch_dice(coarse_core, target_patch)
                lprs_patch_dice = _patch_dice(final_core, target_patch)
            totals["heads"] += float(loss_heads.detach().item())
            totals["base"] += float(loss_base.detach().item())
            totals["final"] += float(loss_final.detach().item())
            totals["boundary"] += float(loss_boundary.detach().item())
            totals["route"] += float(loss_route.detach().item())
            totals["kl"] += float(loss_kl.detach().item())
            totals["total"] += float(loss.detach().item())
            totals["gpps_patch_dice"] += float(gpps_patch_dice.item())
            totals["lprs_patch_dice"] += float(lprs_patch_dice.item())
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

        denominator = max(1, len(training_loader))
        averages = {key: value / denominator for key, value in totals.items()}
        if scheduler is not None:
            scheduler.step()
        _save_state(lprs, last_path)
        if averages["total"] < best_loss:
            best_loss = averages["total"]
            _save_state(lprs, best_path)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in averages.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"LPRS epoch {epoch}: train_loss={averages['total']:.6f}, "
            f"patch_dice={averages['lprs_patch_dice']:.6f}"
        )

    selected = _selected_checkpoint(output_dir, "lprs", config.checkpoint_selection)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    shutil.copy2(selected, output_dir / "lprs.pth")
    _write_json(output_dir / "lprs_history.json", history)
    return output_dir / "lprs.pth"


def train_two_stage(
    gpps_loader: DataLoader,
    config: TrainingConfig,
    lprs_loader: Optional[DataLoader] = None,
) -> Tuple[Path, Path]:
    """Train GPPS followed by LPRS without reading any evaluation split."""

    config.validate()
    seed_everything(config.seed, deterministic=config.deterministic)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "training_config.json", asdict(config))

    gpps_path = output_dir / "gpps.pth"
    gpps_was_reused = config.resume and gpps_path.is_file()
    if not gpps_was_reused:
        gpps_path = train_gpps(gpps_loader, config)
    else:
        print(f"Reusing completed GPPS checkpoint: {gpps_path}")

    lprs_path = output_dir / "lprs.pth"
    lprs_was_reused = config.resume and gpps_was_reused and lprs_path.is_file()
    if not lprs_was_reused:
        # Make LPRS reproducible whether GPPS was trained now or reused.
        seed_everything(config.seed + 1, deterministic=config.deterministic)
        active_lprs_loader = lprs_loader if lprs_loader is not None else gpps_loader
        lprs_path = train_lprs(active_lprs_loader, config, gpps_path)
    else:
        print(f"Reusing completed LPRS checkpoint: {lprs_path}")

    selection = {
        "checkpoint_selection": config.checkpoint_selection,
        "gpps": str(gpps_path),
        "lprs": str(lprs_path),
        "uses_evaluation_data_for_training_or_selection": False,
    }
    _write_json(output_dir / "selected_checkpoints.json", selection)
    return gpps_path, lprs_path


__all__ = [
    "TrainingConfig",
    "dice_loss",
    "make_data_generator",
    "sample_high_value_patches",
    "seed_data_worker",
    "seed_everything",
    "train_gpps",
    "train_lprs",
    "train_two_stage",
]

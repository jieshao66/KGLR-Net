"""End-to-end inference pipeline for GPPS followed by LPRS."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .models import (
    GaussianSlidingWindowInferer,
    GlobalPriorPromptingSegmenter,
    LocalPriorRefinementSegmenter,
    SignalPriorExtractor,
    get_signal_domain,
    safe_probability_map,
)
from .runtime import load_checkpoint


class KGLRPipeline(nn.Module):
    """Compose the global prior-prompting and local prior-refinement stages."""

    def __init__(
        self,
        dataset: str,
        sam_checkpoint: Path | str,
        gpps_checkpoint: Path | str,
        lprs_checkpoint: Path | str,
        *,
        model_type: str = "vit_b",
        lora_rank: int = 4,
        patch_size: int = 56,
        overlap: float = 0.5,
        sigma_scale: float = 0.5,
        sw_batch_size: int = 1,
        use_halo_context: bool = True,
        halo_size: int = 8,
        use_mask_addition: bool = False,
    ) -> None:
        super().__init__()
        signal_domain = get_signal_domain(dataset)
        self.gpps = GlobalPriorPromptingSegmenter(
            model_type=model_type,
            checkpoint_path=str(sam_checkpoint),
            lora_rank=lora_rank,
            signal_domain=signal_domain,
        )
        self.lprs = LocalPriorRefinementSegmenter(use_mask_addition=use_mask_addition)
        self.prior_extractor = SignalPriorExtractor(domain=signal_domain)
        self.inferer = GaussianSlidingWindowInferer(
            patch_size=patch_size,
            overlap=overlap,
            sigma_scale=sigma_scale,
            sw_batch_size=sw_batch_size,
            use_halo_context=use_halo_context,
            halo_size=halo_size,
        )

        self.gpps.load_state_dict(load_checkpoint(gpps_checkpoint), strict=True)
        self.lprs.load_state_dict(load_checkpoint(lprs_checkpoint), strict=True)
        for module in (self.gpps, self.lprs, self.prior_extractor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict global and refined masks for an ImageNet-normalized RGB batch."""

        global_mask = safe_probability_map(self.gpps(image))
        texture_prior, intensity_prior = self.prior_extractor(
            image,
            teacher_a_prob=global_mask,
        )
        refined_mask, local_features = self.inferer.infer(
            image,
            global_mask,
            texture_prior,
            intensity_prior,
            self.lprs,
        )
        return {
            "gpps": safe_probability_map(global_mask),
            "kglr": safe_probability_map(refined_mask),
            "local_features": local_features,
        }


__all__ = ["KGLRPipeline"]

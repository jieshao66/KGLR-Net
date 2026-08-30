"""KGLR-Net model package.

The public API exposes the global prior prompting segmenter (GPPS), the
local prior refinement segmenter (LPRS), and the deterministic prior and
sliding-window utilities used by both stages.
"""

from .models import (
    GaussianSlidingWindowInferer,
    GlobalPriorPromptingSegmenter,
    LocalPriorRefinementSegmenter,
    SignalPriorExtractor,
    get_signal_domain,
    safe_probability_map,
)

__all__ = [
    "GaussianSlidingWindowInferer",
    "GlobalPriorPromptingSegmenter",
    "LocalPriorRefinementSegmenter",
    "SignalPriorExtractor",
    "get_signal_domain",
    "safe_probability_map",
]

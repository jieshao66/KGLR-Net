"""Model definitions for KGLR-Net.

This module implements the deterministic modality-specific prior extractor,
the global prior prompting segmenter (GPPS), the local prior refinement
segmenter (LPRS), and Gaussian-weighted sliding-window inference.  The SAM
backbone is provided by the external ``segment_anything`` package; pretrained
weights are intentionally not bundled with this repository.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as cc_label, distance_transform_edt, binary_dilation


__all__ = [
    "GaussianSlidingWindowInferer",
    "GlobalPriorPromptingSegmenter",
    "LocalPriorRefinementSegmenter",
    "SignalPriorExtractor",
    "get_signal_domain",
    "safe_probability_map",
]


def safe_probability_map(tensor, eps=1e-6):
    """Sanitize and clamp a probability map to ``[eps, 1 - eps]``.

    NaN values are replaced by 0.5, infinities are mapped to the appropriate
    boundary, and otherwise valid probabilities are left unchanged apart from
    the final numerical clamp.
    """
    tensor = torch.nan_to_num(tensor, nan=0.5, posinf=1.0 - eps, neginf=eps)

    return torch.clamp(tensor, min=eps, max=1.0 - eps)


def safe_finite_tensor(tensor, fill_value=0.0, clamp_min=None, clamp_max=None):
    """Replace non-finite feature values without otherwise normalizing them."""
    tensor = torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    if clamp_min is not None or clamp_max is not None:
        tensor = torch.clamp(tensor, min=clamp_min, max=clamp_max)

    return tensor


def safe_simplex_weights(weights, dim=1, eps=1e-6):
    """Safely normalize non-negative routing weights on a simplex.

    Non-finite entries are replaced by zero.  A location whose complete weight
    vector is invalid falls back to a uniform distribution.
    """
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    weights = torch.clamp(weights, min=0.0)

    denom = weights.sum(dim=dim, keepdim=True)

    num_classes = weights.shape[dim]

    uniform = torch.full_like(weights, 1.0 / float(num_classes))

    normalized = weights / torch.clamp(denom, min=eps)

    return torch.where(denom > eps, normalized, uniform)


# =====================================================================
# 2-D phase congruency with Log-Gabor filters
# =====================================================================
class PhaseCongruency2D_PyTorch(nn.Module):
    """Batched 2-D phase congruency using Log-Gabor filters.

    The default configuration uses three scales and four orientations and
    returns the maximum and minimum phase-congruency moments.
    """

    def __init__(self, n_scales=3, n_orientations=4, min_wave_length=3, mult=2.0, sigma_on_f=0.55):
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.min_wave_length = min_wave_length
        self.mult = mult
        self.sigma_on_f = sigma_on_f
        self.epsilon = 1e-4
        # Filters depend on the input size and are not trainable.  The
        # non-persistent buffer follows device transfers without entering a
        # checkpoint.
        self.register_buffer("filters", torch.empty(0), persistent=False)

    def _create_log_gabor_filters(self, H, W, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=device, dtype=dtype),
            torch.linspace(-0.5, 0.5, W, device=device, dtype=dtype),
            indexing="ij",
        )
        radius = torch.sqrt(x**2 + y**2)
        radius[H // 2, W // 2] = 1.0
        theta = torch.atan2(-y, x)

        filters = []
        for o in range(self.n_orientations):
            angl = o * math.pi / self.n_orientations
            ds = torch.sin(theta) * math.cos(angl) - torch.cos(theta) * math.sin(angl)
            dc = torch.cos(theta) * math.cos(angl) + torch.sin(theta) * math.sin(angl)
            dtheta = torch.abs(torch.atan2(ds, dc))
            spread = torch.exp((-(dtheta**2)) / (2 * 1.5**2))

            for s in range(self.n_scales):
                wavelength = self.min_wave_length * (self.mult**s)
                fo = 1.0 / wavelength
                log_gabor = torch.exp(
                    (-((torch.log(radius / fo)) ** 2)) / (2 * math.log(self.sigma_on_f) ** 2)
                )
                log_gabor[H // 2, W // 2] = 0.0

                filter_2d = log_gabor * spread
                filter_2d = torch.fft.ifftshift(filter_2d)
                filters.append(filter_2d)

        return torch.stack(filters, dim=0).unsqueeze(0)

    def forward(self, img):
        B, C, H, W = img.shape
        device = img.device

        if C != 1:
            raise ValueError(
                "PhaseCongruency2D_PyTorch expects a single-channel image, " f"but received C={C}."
            )

        # Rebuild the filters when shape, device, or dtype changes.
        need_rebuild = (
            self.filters.numel() == 0
            or self.filters.shape[-2:] != (H, W)
            or self.filters.device != device
            or self.filters.dtype != img.dtype
        )

        if need_rebuild:
            self.filters = self._create_log_gabor_filters(H=H, W=W, device=device, dtype=img.dtype)

        img_fft = torch.fft.fft2(img)
        filtered_fft = img_fft * self.filters
        filtered_spatial = torch.fft.ifft2(filtered_fft)

        EO_real = filtered_spatial.real
        EO_imag = filtered_spatial.imag

        An = torch.sqrt(EO_real**2 + EO_imag**2)

        EO_real = EO_real.view(B, self.n_orientations, self.n_scales, H, W)
        EO_imag = EO_imag.view(B, self.n_orientations, self.n_scales, H, W)
        An = An.view(B, self.n_orientations, self.n_scales, H, W)

        sum_E = torch.sum(EO_real, dim=2)
        sum_O = torch.sum(EO_imag, dim=2)
        sum_An = torch.sum(An, dim=2)

        Energy = torch.sqrt(sum_E**2 + sum_O**2)
        PC = Energy / (sum_An + self.epsilon)

        # Preserve the input device and dtype, including mixed-precision use.
        pc_x = PC.new_zeros((B, H, W))

        pc_y = PC.new_zeros((B, H, W))

        pc_xy = PC.new_zeros((B, H, W))

        for o in range(self.n_orientations):
            angl = o * math.pi / self.n_orientations
            c = math.cos(angl)
            s = math.sin(angl)
            pc_o = PC[:, o, :, :]

            pc_x += (pc_o * c) ** 2
            pc_y += (pc_o * s) ** 2
            pc_xy += (pc_o * c) * (pc_o * s)

        pc_x = pc_x / self.n_orientations
        pc_y = pc_y / self.n_orientations
        pc_xy = pc_xy / self.n_orientations

        trace = pc_x + pc_y
        diff = pc_x - pc_y
        sqrt_term = torch.sqrt(diff**2 + 4 * pc_xy**2)

        M_max = 0.5 * (trace + sqrt_term)
        M_min = 0.5 * (trace - sqrt_term)

        return torch.stack([M_max, M_min], dim=1)


# =====================================================================
# Modality-specific deterministic prior extractor
# =====================================================================
def get_signal_domain(dataset_name):
    """Map a dataset name to its modality-specific prior domain.

    ``skin`` covers ISIC2018 and HAM10000; ``polyp`` covers Kvasir-SEG and
    CVC-ClinicDB.  Keeping this mapping central ensures that training,
    evaluation, cross-dataset evaluation, and visualization use identical
    prior definitions.
    """
    if dataset_name in {"ISIC2018", "HAM10000"}:
        return "skin"

    if dataset_name in {"Kvasir-SEG", "CVC-ClinicDB"}:
        return "polyp"

    # Preserve the historical skin-domain fallback for unknown datasets.
    return "skin"


class SignalPriorExtractor(nn.Module):
    def __init__(self, input_is_imagenet_normalized=True, domain="skin"):
        super().__init__()

        domain = str(domain).lower()
        if domain not in {"skin", "polyp"}:
            raise ValueError(
                "SignalPriorExtractor domain must be 'skin' or 'polyp', "
                f"but received {domain!r}."
            )
        self.domain = domain

        # Dataset images are ImageNet-normalized by default.  Priors are
        # computed after restoring physical RGB intensities in [0, 1].
        self.input_is_imagenet_normalized = input_is_imagenet_normalized

        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

        # Four Haar-DWT channels: LL, LH, HL, and HH.
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        dwt_weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.dwt_conv = nn.Conv2d(1, 4, kernel_size=2, stride=1, padding=0, bias=False)
        self.dwt_conv.weight = nn.Parameter(dwt_weight, requires_grad=False)

        # Dermoscopic Gabor texture bank.  The polyp branch instead uses its
        # modality-specific multiscale boundary gradients.
        gabor_weight = self._generate_gabor_kernels()
        self.gabor_conv = nn.Conv2d(1, 4, kernel_size=7, stride=1, padding=3, bias=False)
        self.gabor_conv.weight = nn.Parameter(gabor_weight, requires_grad=False)

        # Fixed dermoscopic fractional-order texture-complexity prior.
        v = 0.5
        frac_kernel = torch.tensor(
            [[-v / 4, -v / 2, -v / 4], [-v / 2, 3 * v, -v / 2], [-v / 4, -v / 2, -v / 4]]
        )
        frac_weight = frac_kernel.view(1, 1, 3, 3)
        self.frac_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.frac_conv.weight = nn.Parameter(frac_weight, requires_grad=False)

        # Native PyTorch phase-congruency operator.
        self.pc_extractor = PhaseCongruency2D_PyTorch(n_scales=3, n_orientations=4)

        # Fixed Sobel kernels for multiscale polyp boundary gradients.
        sobel_x = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            / 8.0
        )
        sobel_y = (
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            / 8.0
        )
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _generate_gabor_kernels(self, ksize=7, sigma=1.5, lambd=3.0, gamma=0.5):
        kernels = []
        for theta in [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]:
            half_k = ksize // 2
            y, x = torch.meshgrid(
                torch.arange(-half_k, half_k + 1), torch.arange(-half_k, half_k + 1), indexing="ij"
            )
            x_theta = x * math.cos(theta) + y * math.sin(theta)
            y_theta = -x * math.sin(theta) + y * math.cos(theta)
            gb = torch.exp(
                -(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)
            ) * torch.cos(2 * math.pi * x_theta / lambd)
            kernels.append(gb)
        return torch.stack(kernels, dim=0).unsqueeze(1)

    def to_physical_rgb(self, img):
        """Restore an input image to the physical ``[0, 1]`` RGB range."""
        if self.input_is_imagenet_normalized:
            img = img * self.imagenet_std + self.imagenet_mean

        return torch.clamp(img, min=0.0, max=1.0)

    def _get_dark_channel(self, img, patch_size=7):
        min_rgb, _ = torch.min(img, dim=1, keepdim=True)
        pad = patch_size // 2
        return -F.max_pool2d(-min_rgb, kernel_size=patch_size, stride=1, padding=pad)

    def _get_pc_moments(self, gray):
        """Extract the maximum and minimum phase-congruency moments."""
        pc_moments = self.pc_extractor(gray)
        return torch.clamp(pc_moments, 0.0, 1.0)

    @staticmethod
    def _avg_pool_same(x, kernel_size):
        pad = kernel_size // 2
        return F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def _sobel_magnitude(self, x):
        gx = F.conv2d(x, self.sobel_x.to(device=x.device, dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(device=x.device, dtype=x.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    def _polyp_region_priors(self, img):
        """Compute polyp color, local-contrast, and tissue priors."""
        eps = 1e-6
        R = img[:, 0:1]
        G = img[:, 1:2]
        B = img[:, 2:3]

        max_rgb = torch.max(torch.max(R, G), B)
        min_rgb = torch.min(torch.min(R, G), B)
        rgb_sum = R + G + B + eps

        brightness = max_rgb
        saturation = (max_rgb - min_rgb) / (max_rgb + eps)
        saturation = torch.clamp(saturation, 0.0, 1.0)

        # Normalized red opponency is robust to brightness and yellow mucus.
        r_norm = R / rgb_sum
        g_norm = G / rgb_sum
        b_norm = B / rgb_sum
        red_opponent = r_norm - torch.max(g_norm, b_norm)
        red_opponent = torch.clamp((red_opponent + 0.20) / 0.40, 0.0, 1.0)

        polyp_color_saliency = torch.clamp(0.70 * red_opponent + 0.30 * saturation, 0.0, 1.0)

        # Suppress black endoscope borders and very dark cavities.
        valid_tissue = torch.sigmoid(80.0 * (brightness - 0.08))

        # Use soft specular weights to avoid hard-threshold boundary loss.
        specular = torch.sigmoid(30.0 * (brightness - 0.80)) * torch.sigmoid(
            24.0 * (0.35 - saturation)
        )

        tissue_reliability = torch.clamp(valid_tissue * (1.0 - specular), 0.0, 1.0)

        # Measure chromatic abnormality relative to multiscale neighborhoods.
        mean_color_15 = self._avg_pool_same(polyp_color_saliency, 15)
        mean_color_31 = self._avg_pool_same(polyp_color_saliency, 31)
        mean_sat_15 = self._avg_pool_same(saturation, 15)
        mean_sat_31 = self._avg_pool_same(saturation, 31)

        local_chroma_contrast = (
            0.35 * torch.abs(polyp_color_saliency - mean_color_15)
            + 0.35 * torch.abs(polyp_color_saliency - mean_color_31)
            + 0.15 * torch.abs(saturation - mean_sat_15)
            + 0.15 * torch.abs(saturation - mean_sat_31)
        )

        # Rescale the small contrast response into a stable [0, 1] range.
        local_chroma_contrast = torch.clamp(3.0 * local_chroma_contrast, 0.0, 1.0)

        polyp_color_saliency = polyp_color_saliency * tissue_reliability
        local_chroma_contrast = local_chroma_contrast * tissue_reliability

        s_int_polyp = torch.cat(
            [polyp_color_saliency, local_chroma_contrast, tissue_reliability], dim=1
        )

        # Residual reliability weighting preserves low-contrast true boundaries
        # while suppressing borders, highlights, and ordinary mucosal folds.
        structure_weight_raw = tissue_reliability * (0.5 + 0.5 * local_chroma_contrast)

        structure_weight = torch.clamp(0.25 + 0.75 * structure_weight_raw, 0.0, 1.0)

        return s_int_polyp, tissue_reliability, local_chroma_contrast, structure_weight

    def _polyp_multiscale_boundary(self, gray, color_saliency, local_contrast, structure_weight):
        """Compute three polyp-aware multiscale boundary gradients."""
        structural_source = torch.clamp(
            0.50 * gray + 0.30 * color_saliency + 0.20 * local_contrast, 0.0, 1.0
        )

        boundary_maps = []
        for kernel_size in (3, 7, 11):
            smoothed = self._avg_pool_same(structural_source, kernel_size)
            grad = self._sobel_magnitude(smoothed)
            grad = torch.clamp(4.0 * grad, 0.0, 1.0)
            boundary_maps.append(grad * structure_weight)

        return torch.cat(boundary_maps, dim=1)

    def _dwt_features(self, gray):
        # --------------------------------------------------
        # Reflect-pad the right and bottom edges before the Haar DWT so that
        # [B, 1, H, W] maps to [B, 4, H, W].
        # --------------------------------------------------
        _, _, H, W = gray.shape
        gray_for_dwt = F.pad(gray, pad=(0, 1, 0, 1), mode="reflect")

        dwt_feat = self.dwt_conv(gray_for_dwt)

        if dwt_feat.shape[-2:] != (H, W):
            raise RuntimeError(
                "Unexpected DWT output size: "
                f"expected {(H, W)}, got {tuple(dwt_feat.shape[-2:])}."
            )

        return dwt_feat

    @staticmethod
    def _norm01(x, eps=1e-6):
        """Min-max normalize each fixed prior map to ``[0, 1]``."""
        lo = x.amin(dim=(-2, -1), keepdim=True)
        hi = x.amax(dim=(-2, -1), keepdim=True)
        return torch.clamp((x - lo) / torch.clamp(hi - lo, min=eps), 0.0, 1.0)

    @staticmethod
    def _fast_avg_pool_same(x, kernel_size):
        pad = kernel_size // 2
        return F.avg_pool2d(
            x, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False
        )

    @torch.no_grad()
    def extract_polyp_rich(self, img):
        """Extract the rich polyp prior used by GPPS and LPRS."""
        img = self.to_physical_rgb(img)
        B, C, H, W = img.shape
        if C != 3:
            raise ValueError(f"The polyp prior expects RGB input, but received C={C}.")

        eps = 1e-6
        R = img[:, 0:1]
        G = img[:, 1:2]
        Bc = img[:, 2:3]
        gray = 0.299 * R + 0.587 * G + 0.114 * Bc
        max_rgb = torch.max(torch.max(R, G), Bc)
        min_rgb = torch.min(torch.min(R, G), Bc)
        rgb_sum = R + G + Bc + eps
        brightness = max_rgb
        saturation = torch.clamp((max_rgb - min_rgb) / torch.clamp(max_rgb, min=eps), 0.0, 1.0)

        r_norm = R / rgb_sum
        g_norm = G / rgb_sum
        b_norm = Bc / rgb_sum
        red_opponent = torch.clamp((r_norm - torch.max(g_norm, b_norm) + 0.20) / 0.40, 0.0, 1.0)
        rg_pink = torch.clamp((R - Bc + 0.15) / 0.45, 0.0, 1.0)
        red_pink_saliency = self._norm01(0.65 * red_opponent + 0.25 * saturation + 0.10 * rg_pink)

        pale_bright_saliency = torch.sigmoid(14.0 * (brightness - 0.45)) * torch.sigmoid(
            12.0 * (0.72 - saturation)
        )
        pale_bright_saliency = self._norm01(pale_bright_saliency)

        yellow_axis = torch.clamp((R + G) * 0.5 - Bc, 0.0, 1.0)
        yellow_white_polyp_saliency = self._norm01(0.65 * yellow_axis + 0.35 * pale_bright_saliency)

        specular_highlight_map = torch.clamp(
            torch.sigmoid(30.0 * (brightness - 0.80)) * torch.sigmoid(24.0 * (0.35 - saturation)),
            0.0,
            1.0,
        )
        dark_lumen_map = torch.clamp(torch.sigmoid(70.0 * (0.09 - brightness)), 0.0, 1.0)
        black_border_map = torch.clamp(torch.sigmoid(90.0 * (0.06 - brightness)), 0.0, 1.0)
        instrument_or_white_artifact_map = torch.clamp(
            torch.sigmoid(24.0 * (brightness - 0.72)) * torch.sigmoid(20.0 * (0.22 - saturation)),
            0.0,
            1.0,
        )
        mucus_yellow_artifact_map = torch.clamp(
            torch.sigmoid(18.0 * (yellow_axis - 0.18))
            * torch.sigmoid(14.0 * (saturation - 0.28))
            * torch.sigmoid(18.0 * (0.88 - brightness)),
            0.0,
            1.0,
        )

        valid_tissue = torch.sigmoid(70.0 * (brightness - 0.08))
        tissue_reliability = torch.clamp(
            valid_tissue
            * (1.0 - 0.85 * specular_highlight_map)
            * (1.0 - 0.95 * dark_lumen_map)
            * (1.0 - 0.95 * black_border_map)
            * (1.0 - 0.55 * instrument_or_white_artifact_map),
            0.0,
            1.0,
        )
        distractor_map = torch.clamp(
            0.30 * specular_highlight_map
            + 0.25 * torch.max(dark_lumen_map, black_border_map)
            + 0.20 * instrument_or_white_artifact_map
            + 0.20 * mucus_yellow_artifact_map
            + 0.05 * (1.0 - valid_tissue),
            0.0,
            1.0,
        )

        local_contrast_maps = []
        for k in (7, 15, 31):
            mean_gray = self._fast_avg_pool_same(gray, k)
            mean_color = self._fast_avg_pool_same(red_pink_saliency, k)
            mean_yellow = self._fast_avg_pool_same(yellow_white_polyp_saliency, k)
            contrast = (
                0.45 * torch.abs(gray - mean_gray)
                + 0.35 * torch.abs(red_pink_saliency - mean_color)
                + 0.20 * torch.abs(yellow_white_polyp_saliency - mean_yellow)
            )
            local_contrast_maps.append(torch.clamp(3.5 * contrast * tissue_reliability, 0.0, 1.0))
        local_contrast_small, local_contrast_mid, local_contrast_large = local_contrast_maps

        structural_source = torch.clamp(
            0.35 * gray
            + 0.25 * red_pink_saliency
            + 0.20 * pale_bright_saliency
            + 0.20 * yellow_white_polyp_saliency,
            0.0,
            1.0,
        )
        boundaries = []
        for k in (3, 7, 11):
            smoothed = self._fast_avg_pool_same(structural_source, k)
            edge = torch.clamp(4.0 * self._sobel_magnitude(smoothed), 0.0, 1.0)
            boundaries.append(edge * tissue_reliability * (1.0 - 0.65 * distractor_map))
        boundary_small, boundary_mid, boundary_large = boundaries
        soft_boundary_support = torch.clamp(
            0.45 * boundary_small + 0.35 * boundary_mid + 0.20 * boundary_large, 0.0, 1.0
        )

        brightness_residual = (
            self._norm01(
                torch.clamp(brightness - self._fast_avg_pool_same(brightness, 31), min=0.0)
            )
            * tissue_reliability
        )
        chroma_residual = (
            self._norm01(
                torch.abs(red_pink_saliency - self._fast_avg_pool_same(red_pink_saliency, 31))
                + torch.abs(
                    yellow_white_polyp_saliency
                    - self._fast_avg_pool_same(yellow_white_polyp_saliency, 31)
                )
            )
            * tissue_reliability
        )
        protrusion_support = torch.clamp(
            0.40 * brightness_residual + 0.35 * local_contrast_mid + 0.25 * soft_boundary_support,
            0.0,
            1.0,
        )
        convex_blob_support = torch.clamp(
            0.50 * protrusion_support
            + 0.30 * local_contrast_large
            + 0.20 * pale_bright_saliency * tissue_reliability,
            0.0,
            1.0,
        )
        fold_or_boundary_distractor = torch.clamp(
            soft_boundary_support
            * (
                1.0
                - self._norm01(red_pink_saliency + yellow_white_polyp_saliency + local_contrast_mid)
            ),
            0.0,
            1.0,
        )

        polyp_objectness = torch.clamp(
            0.26 * red_pink_saliency
            + 0.20 * pale_bright_saliency
            + 0.18 * yellow_white_polyp_saliency
            + 0.18 * local_contrast_mid
            + 0.10 * local_contrast_large
            + 0.08 * protrusion_support,
            0.0,
            1.0,
        )
        polyp_objectness = self._norm01(polyp_objectness)
        polyp_objectness = torch.clamp(
            polyp_objectness * (0.5 + 0.5 * tissue_reliability) * (1.0 - 0.55 * distractor_map),
            0.0,
            1.0,
        )

        objectness_boundary = torch.clamp(self._sobel_magnitude(polyp_objectness) * 5.0, 0.0, 1.0)
        objectness_boundary_product = torch.clamp(
            polyp_objectness * soft_boundary_support * tissue_reliability, 0.0, 1.0
        )
        distractor_suppressed_edge = torch.clamp(
            soft_boundary_support * tissue_reliability * (1.0 - distractor_map), 0.0, 1.0
        )

        rich = {
            "image_rgb": img,
            "gray": gray,
            "brightness": brightness,
            "saturation": saturation,
            "red_pink_saliency": red_pink_saliency,
            "pale_bright_saliency": pale_bright_saliency,
            "yellow_white_polyp_saliency": yellow_white_polyp_saliency,
            "chroma_residual": chroma_residual,
            "brightness_residual": brightness_residual,
            "local_contrast_small": local_contrast_small,
            "local_contrast_mid": local_contrast_mid,
            "local_contrast_large": local_contrast_large,
            "tissue_reliability": tissue_reliability,
            "valid_tissue": valid_tissue,
            "dark_lumen_map": dark_lumen_map,
            "black_border_map": black_border_map,
            "specular_highlight_map": specular_highlight_map,
            "instrument_or_white_artifact_map": instrument_or_white_artifact_map,
            "mucus_yellow_artifact_map": mucus_yellow_artifact_map,
            "distractor_map": distractor_map,
            "boundary_small": boundary_small,
            "boundary_mid": boundary_mid,
            "boundary_large": boundary_large,
            "soft_boundary_support": soft_boundary_support,
            "objectness_boundary": objectness_boundary,
            "objectness_boundary_product": objectness_boundary_product,
            "distractor_suppressed_edge": distractor_suppressed_edge,
            "convex_blob_support": convex_blob_support,
            "protrusion_support": protrusion_support,
            "fold_or_boundary_distractor": fold_or_boundary_distractor,
            "polyp_objectness": polyp_objectness,
        }

        cleaned = {}
        for k, v in rich.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4 and k != "image_rgb":
                cleaned[k] = safe_finite_tensor(v, fill_value=0.0, clamp_min=0.0, clamp_max=1.0)
            else:
                cleaned[k] = v
        return cleaned

    @torch.no_grad()
    def project_polyp_rich_to_compact(self, rich, teacher_a_prob=None):
        """Project the GPPS-guided rich prior to the 9+3 LPRS interface."""
        objectness = rich["polyp_objectness"]
        tissue = rich["tissue_reliability"]
        distractor = rich["distractor_map"]
        boundary_small = rich["boundary_small"]
        boundary_mid = rich["boundary_mid"]
        boundary_large = rich["boundary_large"]
        objectness_boundary = rich["objectness_boundary"]
        distractor_suppressed_edge = rich["distractor_suppressed_edge"]
        protrusion_or_convex = torch.clamp(
            0.55 * rich["protrusion_support"] + 0.45 * rich["convex_blob_support"], 0.0, 1.0
        )
        objectness_boundary_product = rich["objectness_boundary_product"]

        if teacher_a_prob is None:
            teacher_a_boundary_band = objectness_boundary
            teacher_a_prior_disagreement = torch.zeros_like(objectness)
        else:
            teacher_a_prob = safe_probability_map(teacher_a_prob.detach())
            if teacher_a_prob.shape[-2:] != objectness.shape[-2:]:
                teacher_a_prob = F.interpolate(
                    teacher_a_prob, size=objectness.shape[-2:], mode="bilinear", align_corners=False
                )
            teacher_a_boundary_band = torch.clamp(
                self._sobel_magnitude(teacher_a_prob) * 5.0, 0.0, 1.0
            )
            teacher_a_prior_disagreement = torch.clamp(
                objectness * (1.0 - teacher_a_prob) + teacher_a_prob * (1.0 - objectness), 0.0, 1.0
            ) * (0.5 + 0.5 * tissue)

        s_int = torch.cat([objectness, tissue, distractor], dim=1)
        s_tex = torch.cat(
            [
                boundary_small,
                boundary_mid,
                boundary_large,
                objectness_boundary,
                teacher_a_boundary_band,
                teacher_a_prior_disagreement,
                distractor_suppressed_edge,
                protrusion_or_convex,
                objectness_boundary_product,
            ],
            dim=1,
        )
        s_tex = safe_finite_tensor(s_tex, fill_value=0.0, clamp_min=0.0, clamp_max=1.0)
        s_int = safe_finite_tensor(s_int, fill_value=0.0, clamp_min=0.0, clamp_max=1.0)
        return s_tex, s_int

    @torch.no_grad()
    def forward(self, img, teacher_a_prob=None, return_rich=False):
        # Restore physical RGB intensities before computing any prior.
        img_physical = self.to_physical_rgb(img)
        B, C, H, W = img_physical.shape
        if C != 3:
            raise ValueError("SignalPriorExtractor expects RGB input, " f"but received C={C}.")
        gray = (
            0.299 * img_physical[:, 0:1]
            + 0.587 * img_physical[:, 1:2]
            + 0.114 * img_physical[:, 2:3]
        )

        if self.domain == "skin":
            dwt_feat = self._dwt_features(gray)
            dwt_ll = dwt_feat[:, 0:1]
            dwt_high = dwt_feat[:, 1:4]
            pc_moments = self._get_pc_moments(gray)
            gabor_feat = self.gabor_conv(gray)
            dc_feat = self._get_dark_channel(img_physical)
            frac_feat = self.frac_conv(gray)
            s_tex = torch.cat([gabor_feat, dwt_high, pc_moments], dim=1)
            s_int = torch.cat([dc_feat, dwt_ll, frac_feat], dim=1)
            rich = None
        else:
            rich = self.extract_polyp_rich(img)
            s_tex, s_int = self.project_polyp_rich_to_compact(
                rich=rich, teacher_a_prob=teacher_a_prob
            )

        if s_tex.shape[1] != 9 or s_int.shape[1] != 3:
            raise RuntimeError(
                "SignalPriorExtractor produced an invalid channel count: "
                f"s_tex={s_tex.shape[1]}, s_int={s_int.shape[1]}; "
                "expected s_tex=9 and s_int=3."
            )
        s_tex = safe_finite_tensor(s_tex, fill_value=0.0)
        s_int = safe_finite_tensor(s_int, fill_value=0.0)
        if return_rich:
            return s_tex, s_int, rich
        return s_tex, s_int


# =====================================================================
# Core convolution and attention blocks
# =====================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Match the residual branch width.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out += residual
        return self.relu(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(0.1)
        self.in_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        q = self.query(x_norm).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x_norm).view(B, -1, H * W)
        v = self.value(x_norm).view(B, -1, H * W)
        scale = math.sqrt(self.in_channels // 8)
        attn = F.softmax(torch.bmm(q, k) / scale, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, out_dim):
        super().__init__()
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q_conv = nn.Conv2d(q_dim, out_dim // 8, 1)
        self.k_conv = nn.Conv2d(kv_dim, out_dim // 8, 1)
        self.v_conv = nn.Conv2d(kv_dim, out_dim, 1)
        self.out_conv = nn.Sequential(nn.Conv2d(out_dim, out_dim, 1), nn.Dropout2d(0.1))
        self.out_dim = out_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, q, k_v):
        B, C_q, H, W = q.shape
        _, C_kv, _, _ = k_v.shape
        if q.shape[0] != k_v.shape[0]:
            raise ValueError(
                "CrossAttention inputs have different batch sizes: "
                f"q={tuple(q.shape)}, "
                f"k_v={tuple(k_v.shape)}."
            )

        if q.shape[-2:] != k_v.shape[-2:]:
            raise ValueError(
                "CrossAttention inputs have different spatial sizes: "
                f"q={tuple(q.shape[-2:])}, "
                f"k_v={tuple(k_v.shape[-2:])}."
            )
        q_norm = (
            self.norm_q(q.view(B, C_q, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C_q, H, W)
        )
        kv_norm = (
            self.norm_kv(k_v.view(B, C_kv, -1).permute(0, 2, 1))
            .permute(0, 2, 1)
            .view(B, C_kv, H, W)
        )
        q_flat = self.q_conv(q_norm).view(B, -1, H * W).permute(0, 2, 1)
        k_flat = self.k_conv(kv_norm).view(B, -1, H * W)
        v_flat = self.v_conv(kv_norm).view(B, -1, H * W)
        scale = math.sqrt(self.out_dim // 8)
        attn = F.softmax(torch.bmm(q_flat, k_flat) / scale, dim=-1)
        out = torch.bmm(v_flat, attn.permute(0, 2, 1)).view(B, -1, H, W)
        return q + self.gamma * self.out_conv(out)


class DynamicCarving(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels, channels * channels * 3 * 3 + channels),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, Dd_l, D_l_prime):
        N, C, H, W = Dd_l.shape
        v_l = F.adaptive_avg_pool2d(D_l_prime, 1).view(N, C)
        params = self.mlp(v_l)
        weight_size = C * C * 3 * 3
        weight = params[:, :weight_size].contiguous().view(N * C, C, 3, 3)
        bias = params[:, weight_size:].contiguous().view(N * C)
        x_reshaped = Dd_l.contiguous().view(1, N * C, H, W)
        out = F.conv2d(x_reshaped, weight, bias=bias, padding=1, groups=N)
        out = out.view(N, C, H, W)
        return self.relu(self.norm(out) + Dd_l)


class GatedProtectiveFusion(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.int_gate_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1), nn.BatchNorm2d(out_c), nn.Sigmoid()
        )
        self.final_fuse = DoubleConv(out_c * 2, out_c)

    def forward(self, carved_tex, carved_int):
        int_mask = self.int_gate_conv(carved_int)
        carved_tex_protected = carved_tex + (int_mask * carved_int)
        Dd_l_prime_cat = torch.cat([carved_tex_protected, carved_int], dim=1)
        return self.final_fuse(Dd_l_prime_cat)


def reparameterize_feature(mu, log_var, training, tau=0.1):
    """Apply VAE-style feature reparameterization.

    Training uses stochastic samples for robustness; evaluation uses the mean
    for deterministic inference.  Clamping ``log_var`` prevents overflow.
    """
    log_var = torch.clamp(log_var, min=-10.0, max=10.0)

    if training:
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(mu) * std * tau
    else:
        z = mu

    return z, log_var


# =====================================================================
# Full-attention dual-stream encoder and decoder blocks
# =====================================================================
class FullAttn_EncoderBlock(nn.Module):
    def __init__(self, f_in, s_tex_in, s_int_in, out_c):
        super().__init__()
        self.f_conv = DoubleConv(f_in, out_c)
        self.s_conv_tex = DoubleConv(s_tex_in, out_c)
        self.s_conv_int = DoubleConv(s_int_in, out_c)

        self.f_self_attn = SelfAttention(out_c)
        self.s_spatial_attn_tex = SpatialAttention()
        self.s_spatial_attn_int = SpatialAttention()

        self.f_channel_attn = ChannelAttention(out_c)
        self.s_channel_attn_tex = ChannelAttention(out_c)
        self.s_channel_attn_int = ChannelAttention(out_c)

        self.conv_mu_tex = nn.Conv2d(out_c, out_c, 1)
        self.conv_lv_tex = nn.Conv2d(out_c, out_c, 1)
        self.conv_mu_int = nn.Conv2d(out_c, out_c, 1)
        self.conv_lv_int = nn.Conv2d(out_c, out_c, 1)

        self.cross_attn_tex = CrossAttention(q_dim=out_c, kv_dim=out_c, out_dim=out_c)
        self.cross_attn_int = CrossAttention(q_dim=out_c, kv_dim=out_c, out_dim=out_c)

    def forward(self, f, s_tex, s_int):
        f_feat = self.f_channel_attn(self.f_self_attn(self.f_conv(f)))
        s_tex_feat = self.s_channel_attn_tex(self.s_spatial_attn_tex(self.s_conv_tex(s_tex)))
        s_int_feat = self.s_channel_attn_int(self.s_spatial_attn_int(self.s_conv_int(s_int)))

        mu_t = self.conv_mu_tex(s_tex_feat)
        lv_t = self.conv_lv_tex(s_tex_feat)

        mu_i = self.conv_mu_int(s_int_feat)
        lv_i = self.conv_lv_int(s_int_feat)

        Z_tex, lv_t = reparameterize_feature(mu=mu_t, log_var=lv_t, training=self.training, tau=0.1)

        Z_int, lv_i = reparameterize_feature(mu=mu_i, log_var=lv_i, training=self.training, tau=0.1)

        D_l_prime_tex = self.cross_attn_tex(q=f_feat, k_v=Z_tex)
        D_l_prime_int = self.cross_attn_int(q=f_feat, k_v=Z_int)

        return (
            f_feat,
            s_tex_feat,
            s_int_feat,
            D_l_prime_tex,
            D_l_prime_int,
            (mu_t, lv_t, mu_i, lv_i),
        )


class FullAttn_DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.f_conv = DoubleConv(in_c + skip_c, out_c)
        self.s_conv_tex = DoubleConv(in_c + skip_c, out_c)
        self.s_conv_int = DoubleConv(in_c + skip_c, out_c)

        self.f_self_attn = SelfAttention(out_c)
        self.s_spatial_attn_tex = SpatialAttention()
        self.s_spatial_attn_int = SpatialAttention()

        self.f_channel_attn = ChannelAttention(out_c)
        self.s_channel_attn_tex = ChannelAttention(out_c)
        self.s_channel_attn_int = ChannelAttention(out_c)

        self.conv_mu_tex = nn.Conv2d(out_c, out_c, 1)
        self.conv_lv_tex = nn.Conv2d(out_c, out_c, 1)
        self.conv_mu_int = nn.Conv2d(out_c, out_c, 1)
        self.conv_lv_int = nn.Conv2d(out_c, out_c, 1)

        self.cross_attn_tex = CrossAttention(q_dim=out_c, kv_dim=out_c, out_dim=out_c)
        self.cross_attn_int = CrossAttention(q_dim=out_c, kv_dim=out_c, out_dim=out_c)

        self.dynamic_carving_tex = DynamicCarving(out_c)
        self.dynamic_carving_int = DynamicCarving(out_c)
        self.fuse_carved = GatedProtectiveFusion(out_c)

    def forward(
        self, f_up, s_t_up, s_i_up, f_skip, s_t_skip, s_i_skip, D_l_prime_tex, D_l_prime_int
    ):
        f_feat = self.f_channel_attn(
            self.f_self_attn(self.f_conv(torch.cat([self.up(f_up), f_skip], dim=1)))
        )
        s_t_feat = self.s_channel_attn_tex(
            self.s_spatial_attn_tex(self.s_conv_tex(torch.cat([self.up(s_t_up), s_t_skip], dim=1)))
        )
        s_i_feat = self.s_channel_attn_int(
            self.s_spatial_attn_int(self.s_conv_int(torch.cat([self.up(s_i_up), s_i_skip], dim=1)))
        )

        mu_t = self.conv_mu_tex(s_t_feat)
        lv_t = self.conv_lv_tex(s_t_feat)

        mu_i = self.conv_mu_int(s_i_feat)
        lv_i = self.conv_lv_int(s_i_feat)

        Z_tex_dec, lv_t = reparameterize_feature(
            mu=mu_t, log_var=lv_t, training=self.training, tau=0.1
        )

        Z_int_dec, lv_i = reparameterize_feature(
            mu=mu_i, log_var=lv_i, training=self.training, tau=0.1
        )

        Dd_l_tex = self.cross_attn_tex(q=Z_tex_dec, k_v=f_feat)
        Dd_l_int = self.cross_attn_int(q=Z_int_dec, k_v=f_feat)

        carved_tex = self.dynamic_carving_tex(Dd_l_tex, D_l_prime_tex)
        carved_int = self.dynamic_carving_int(Dd_l_int, D_l_prime_int)

        return (
            f_feat,
            s_t_feat,
            s_i_feat,
            self.fuse_carved(carved_tex, carved_int),
            (mu_t, lv_t, mu_i, lv_i),
        )


# =====================================================================
# Compact atrous spatial pyramid pooling
# =====================================================================
class MiniASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # The global-average-pooling branch uses GroupNorm so it remains valid
        # for a batch size of one.
        self.b5_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        b5 = F.adaptive_avg_pool2d(x, 1)
        b5 = self.b5_conv(b5)
        b5 = F.interpolate(b5, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.conv_out(torch.cat([b1, b2, b3, b4, b5], dim=1))


# =====================================================================
# Base-mask generator
# =====================================================================
class AdvancedMaskGenerator(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        # Squeeze-and-excitation channel attention.
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Progressive residual feature reduction.
        self.decoder = nn.Sequential(
            DoubleConv(in_channels, in_channels // 2),  # 64 -> 32
            DoubleConv(in_channels // 2, in_channels // 4),  # 32 -> 16
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),  # 16 -> 1
        )

    def forward(self, f):
        # Channel-attention enhancement.
        f_se = f * self.se(f)
        # Progressive decoding.
        out = self.decoder(f_se)
        return out


# =====================================================================
# Local prior refinement segmenter (LPRS)
# =====================================================================
class LocalPriorRefinementSegmenter(nn.Module):
    """Refine GPPS masks on local patches using calibrated image priors."""

    def __init__(self, noise_scale=0.1, use_mask_addition=False):
        super().__init__()
        self.use_mask_addition = use_mask_addition
        # Maximum scale of prior-conditioned perturbations.
        self.noise_scale = noise_scale
        # --------------------------------------------------
        # Adaptive calibration of the 12 prior channels.  Residual scaling in
        # [0.5, 1.5] prevents a prior family from being disabled early.
        # --------------------------------------------------
        self.prior_calibrator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(12, 8, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 12, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Predict perturbation scale from the 9 texture and 3 intensity priors.
        self.prior_to_sigma = nn.Sequential(nn.Conv2d(12, 1, kernel_size=1), nn.Sigmoid())

        self.enc1 = FullAttn_EncoderBlock(4, 9, 3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = FullAttn_EncoderBlock(64, 64, 64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = FullAttn_EncoderBlock(128, 128, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = FullAttn_EncoderBlock(256, 256, 256, 512)

        self.dec3 = FullAttn_DecoderBlock(in_c=512, skip_c=256, out_c=256)
        self.dec2 = FullAttn_DecoderBlock(in_c=256, skip_c=128, out_c=128)
        self.dec1 = FullAttn_DecoderBlock(in_c=128, skip_c=64, out_c=64)

        # Multiscale feature aggregation.
        self.final_feat = MiniASPP(in_channels=64, out_channels=64)
        # Disagreement-aware GPPS/LPRS confidence gate: 64 refinement feature
        # channels, two probabilities, and their absolute difference.
        self.conf_head = nn.Sequential(
            nn.Conv2d(67, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Learned base-mask generator.
        self.base_proj = AdvancedMaskGenerator(in_channels=64)

        # Five independent expert heads.
        in_h = 1 + 1 + 64 + 16  # m_a_patch + m_b_base + m_b_feat + learnable
        # Nonlinear mixture-of-experts predictions.
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_h, in_h // 2, 1, bias=False),
                    nn.BatchNorm2d(in_h // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_h // 2, 1, 1),
                )
                for _ in range(5)
            ]
        )
        self.learnable_tensor = nn.Parameter(torch.randn(1, 16, 1, 1))

        # Pixelwise routing from 64 feature channels to five expert weights.
        self.head_router = nn.Conv2d(64, 5, 1)

    def forward(self, x_patch, m_a_patch, s_tex_patch, s_int_patch, return_aux=False):
        """Refine a local patch and return expert and fused predictions.

        The default central patch is 56 x 56.  Halo context may enlarge the
        input, but both spatial dimensions must remain divisible by eight.
        """

        # --------------------------------------------------
        # Validate all four aligned input streams before attention is applied.
        # --------------------------------------------------
        input_tensors = {
            "x_patch": x_patch,
            "m_a_patch": m_a_patch,
            "s_tex_patch": s_tex_patch,
            "s_int_patch": s_int_patch,
        }

        expected_channels = {"x_patch": 3, "m_a_patch": 1, "s_tex_patch": 9, "s_int_patch": 3}

        for tensor_name, tensor in input_tensors.items():
            if tensor.ndim != 4:
                raise ValueError(
                    f"{tensor_name} must be a four-dimensional [B, C, H, W] "
                    f"tensor, but has shape {tuple(tensor.shape)}."
                )

        B, _, H, W = x_patch.shape

        for tensor_name, tensor in input_tensors.items():
            if tensor.shape[0] != B:
                raise ValueError(
                    f"{tensor_name} has a different batch size from x_patch: "
                    f"x_patch={B}, "
                    f"{tensor_name}={tensor.shape[0]}."
                )

            if tensor.shape[1] != expected_channels[tensor_name]:
                raise ValueError(
                    f"{tensor_name} has an invalid channel count: expected "
                    f"{expected_channels[tensor_name]}, got {tensor.shape[1]}."
                )

            if tensor.shape[-2:] != (H, W):
                raise ValueError(
                    "All four LPRS inputs must have identical spatial sizes. "
                    f"x_patch={(H, W)}, "
                    f"{tensor_name}={tuple(tensor.shape[-2:])}."
                )

            if tensor.device != x_patch.device:
                raise ValueError(
                    f"{tensor_name} is not on the same device as x_patch: "
                    f"x_patch={x_patch.device}, "
                    f"{tensor_name}={tensor.device}."
                )

        # B, _, H, W = x_patch.shape

        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(
                "LPRS patch height and width must be divisible by eight, "
                f"but received H={H}, W={W}."
            )

        x_patch = safe_finite_tensor(x_patch, fill_value=0.0)

        m_a_patch = safe_probability_map(m_a_patch)

        s_tex_patch = safe_finite_tensor(s_tex_patch, fill_value=0.0)

        s_int_patch = safe_finite_tensor(s_int_patch, fill_value=0.0)

        # --------------------------------------------------
        # Adaptively calibrate the 12 prior channels.
        # --------------------------------------------------
        prior_all = torch.cat([s_tex_patch, s_int_patch], dim=1)

        prior_gate = self.prior_calibrator(prior_all)

        # Residual gating range: [0.5, 1.5].
        prior_all = prior_all * (0.5 + prior_gate)

        s_tex_patch = prior_all[:, :9, :, :]

        s_int_patch = prior_all[:, 9:, :, :]

        # --------------------------------------------------
        # Estimate perturbation strength from the calibrated priors.
        # --------------------------------------------------
        sigma = self.noise_scale * self.prior_to_sigma(prior_all)

        # Use stochastic perturbations only during training.
        if self.training:
            m_a_noised = m_a_patch + torch.randn_like(m_a_patch) * sigma
            m_a_noised = torch.clamp(m_a_noised, min=0.0, max=1.0)
        else:
            m_a_noised = m_a_patch

        # Fuse RGB and the perturbed coarse mask into a four-channel input.
        # ``use_mask_addition`` selects the corresponding ablation variant.
        if self.use_mask_addition:
            rgb_branch = x_patch + m_a_patch
        else:
            rgb_branch = x_patch

        f_in = torch.cat([rgb_branch, m_a_noised], dim=1)

        # Encode and decode the visual and prior streams.
        f1, st1, si1, D1p_t, D1p_i, v1 = self.enc1(f_in, s_tex_patch, s_int_patch)
        f2, st2, si2, D2p_t, D2p_i, v2 = self.enc2(self.pool1(f1), self.pool1(st1), self.pool1(si1))
        f3, st3, si3, D3p_t, D3p_i, v3 = self.enc3(self.pool2(f2), self.pool2(st2), self.pool2(si2))
        f4, st4, si4, D4p_t, D4p_i, v4 = self.enc4(self.pool3(f3), self.pool3(st3), self.pool3(si3))

        fd3, sd_t3, sd_i3, Dd3p, vd3 = self.dec3(f4, st4, si4, f3, st3, si3, D3p_t, D3p_i)
        fd2, sd_t2, sd_i2, Dd2p, vd2 = self.dec2(Dd3p, sd_t3, sd_i3, f2, st2, si2, D2p_t, D2p_i)
        fd1, sd_t1, sd_i1, Dd1p, vd1 = self.dec1(Dd2p, sd_t2, sd_i2, f1, st1, si1, D1p_t, D1p_i)

        # Multiscale high-level prior feature.
        m_b_feat = self.final_feat(Dd1p)

        # --------------------------------------------------
        # Base prediction.
        # --------------------------------------------------
        m_b_base = safe_probability_map(torch.sigmoid(self.base_proj(m_b_feat)))

        # --------------------------------------------------
        # Shared input to the five expert heads.
        # --------------------------------------------------
        f_out = torch.cat(
            [m_a_patch, m_b_base, m_b_feat, self.learnable_tensor.expand(B, -1, H, W)], dim=1
        )

        # --------------------------------------------------
        # Five expert probability maps.
        # --------------------------------------------------
        preds_b_list = [safe_probability_map(torch.sigmoid(head(f_out))) for head in self.heads]

        preds_b_tensor = torch.stack(preds_b_list, dim=1)

        # --------------------------------------------------
        # Pixelwise expert routing.
        # --------------------------------------------------
        route_logits = self.head_router(m_b_feat)

        route_weights = F.softmax(route_logits, dim=1)

        route_weights = safe_simplex_weights(route_weights, dim=1).unsqueeze(2)

        m_b_expert_fused = torch.sum(preds_b_tensor * route_weights, dim=1)

        m_b_expert_fused = safe_probability_map(m_b_expert_fused)

        # --------------------------------------------------
        # GPPS/LPRS disagreement-aware gating.
        # --------------------------------------------------
        gate_input = torch.cat(
            [m_b_feat, m_a_patch, m_b_expert_fused, torch.abs(m_b_expert_fused - m_a_patch)], dim=1
        )

        w_conf = safe_probability_map(self.conf_head(gate_input))

        # --------------------------------------------------
        # Final GPPS/LPRS fusion.
        # --------------------------------------------------
        m_final = w_conf * m_b_expert_fused + (1.0 - w_conf) * m_a_patch

        m_final = safe_probability_map(m_final)

        # Refined local representation used by sliding-window reconstruction.
        z_sd_pure = (vd1[0] + vd1[2]) / 2.0

        z_sd_pure = safe_finite_tensor(z_sd_pure, fill_value=0.0)

        # Collect VAE distribution parameters for the KL loss.
        mus_lvs = (
            [
                v1[0],
                v1[2],
                v2[0],
                v2[2],
                v3[0],
                v3[2],
                v4[0],
                v4[2],
                vd3[0],
                vd3[2],
                vd2[0],
                vd2[2],
                vd1[0],
                vd1[2],
            ],
            [
                v1[1],
                v1[3],
                v2[1],
                v2[3],
                v3[1],
                v3[3],
                v4[1],
                v4[3],
                vd3[1],
                vd3[3],
                vd2[1],
                vd2[3],
                vd1[1],
                vd1[3],
            ],
        )

        aux = {
            "m_b_base": m_b_base,
            "m_b_expert_fused": m_b_expert_fused,
            "route_weights": route_weights,
            "prior_gate": prior_gate,
        }

        if return_aux:
            return (preds_b_list, mus_lvs, m_final, w_conf, z_sd_pure, aux)

        return (preds_b_list, mus_lvs, m_final, w_conf, z_sd_pure)


# =====================================================================
# Gaussian-weighted sliding-window inference
# =====================================================================
class GaussianSlidingWindowInferer:
    """Reconstruct LPRS masks and features from overlapping local patches."""

    def __init__(
        self,
        patch_size=56,
        overlap=0.5,
        sigma_scale=0.5,
        sw_batch_size=2,
        use_halo_context=False,
        halo_size=8,
    ):
        """Configure patch overlap, Gaussian weighting, and optional halo.

        ``patch_size`` is the central region written to the output canvas.
        When ``use_halo_context`` is true, ``halo_size`` extra pixels on each
        side provide context but are cropped before reconstruction.
        """
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(
                "patch_size must be a positive integer, " f"but received {patch_size}."
            )

        if not (0.0 <= overlap < 1.0):
            raise ValueError(
                "overlap must satisfy 0.0 <= overlap < 1.0, " f"but received {overlap}."
            )

        if sigma_scale <= 0.0:
            raise ValueError(
                "sigma_scale must be greater than zero, " f"but received {sigma_scale}."
            )

        if not isinstance(sw_batch_size, int) or sw_batch_size <= 0:
            raise ValueError(
                "sw_batch_size must be a positive integer, " f"but received {sw_batch_size}."
            )

        if not isinstance(halo_size, int) or halo_size < 0:
            raise ValueError(
                "halo_size must be a non-negative integer, " f"but received {halo_size}."
            )

        self.patch_size = patch_size

        self.use_halo_context = bool(use_halo_context)

        self.requested_halo_size = halo_size

        # Halo size is active only when contextual inference is enabled.
        self.halo_size = halo_size if self.use_halo_context else 0

        self.context_size = self.patch_size + 2 * self.halo_size

        # LPRS applies three successive factor-two downsampling operations.
        if self.context_size % 8 != 0:
            raise ValueError(
                "The effective LPRS input size must be divisible by eight. "
                f"patch_size={self.patch_size}, "
                f"effective_halo_size={self.halo_size}, "
                f"context_size={self.context_size}."
            )

        self.step = int(patch_size * (1.0 - overlap))

        if self.step <= 0:
            raise ValueError(
                "The sliding-window step must be positive, but computed "
                f"step={self.step}. Reduce overlap."
            )

        self.sigma_scale = float(sigma_scale)

        self.sw_batch_size = sw_batch_size

        # Gaussian weights cover only the central valid region.
        self.gaussian_map = self._get_gaussian_window(
            patch_size=self.patch_size, sigma_scale=self.sigma_scale
        )

    @staticmethod
    def _get_gaussian_window(patch_size, sigma_scale):
        coords = torch.arange(patch_size, dtype=torch.float32) - (patch_size - 1) / 2.0

        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

        sigma_pixel = max(float(patch_size) * float(sigma_scale), 1e-6)

        gaussian = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * sigma_pixel**2))

        gaussian = gaussian / gaussian.max()

        # Keep edge weights away from zero for numerical stability.
        gaussian = torch.clamp(gaussian, min=1e-4)

        return gaussian.unsqueeze(0).unsqueeze(0)

    def _get_positions(self, length):
        """Generate window origins and ensure the last touches the edge."""
        if length < self.patch_size:
            raise ValueError(
                f"Input size {length} is smaller than " f"patch_size={self.patch_size}."
            )

        positions = list(range(0, length - self.patch_size + 1, self.step))

        last_position = length - self.patch_size

        if positions[-1] != last_position:
            positions.append(last_position)

        return positions

    @staticmethod
    def _iter_chunks(items, chunk_size):
        for start in range(0, len(items), chunk_size):
            yield items[start : start + chunk_size]

    @staticmethod
    def _center_crop_last_two_dims(tensor, crop_size):
        """Center-crop the final two dimensions of a 4-D or 5-D tensor."""
        height, width = tensor.shape[-2:]

        if crop_size > height or crop_size > width:
            raise ValueError(
                "The center crop exceeds the input dimensions: "
                f"crop_size={crop_size}, "
                f"input={(height, width)}."
            )

        top = (height - crop_size) // 2

        left = (width - crop_size) // 2

        return tensor[..., top : top + crop_size, left : left + crop_size]

    def _pad_for_halo(self, tensor):
        """Apply reflection padding when halo context is enabled."""
        if self.halo_size == 0:
            return tensor

        height, width = tensor.shape[-2:]

        if self.halo_size >= height or self.halo_size >= width:
            raise ValueError(
                "halo_size must be smaller than both input dimensions, "
                f"but received halo_size={self.halo_size}, "
                f"input={(height, width)}."
            )

        return F.pad(
            tensor,
            pad=(self.halo_size, self.halo_size, self.halo_size, self.halo_size),
            mode="reflect",
        )

    @torch.no_grad()
    def infer(self, img_224, m_a_224, s_tex_224, s_int_224, teacher_b_model):
        """Run LPRS over a full image.

        Inputs have shapes ``[B,3,H,W]``, ``[B,1,H,W]``, ``[B,9,H,W]``, and
        ``[B,3,H,W]``.  The returned mask and feature map have shapes
        ``[B,1,H,W]`` and ``[B,64,H,W]``, respectively.
        """

        # --------------------------------------------------
        # Validate the aligned full-image input streams.
        # --------------------------------------------------
        full_inputs = {
            "img_224": img_224,
            "m_a_224": m_a_224,
            "s_tex_224": s_tex_224,
            "s_int_224": s_int_224,
        }

        expected_channels = {"img_224": 3, "m_a_224": 1, "s_tex_224": 9, "s_int_224": 3}

        for tensor_name, tensor in full_inputs.items():
            if tensor.ndim != 4:
                raise ValueError(
                    f"{tensor_name} must be a four-dimensional [B, C, H, W] "
                    f"tensor, but has shape {tuple(tensor.shape)}."
                )

        B, _, H, W = img_224.shape

        for tensor_name, tensor in full_inputs.items():
            if tensor.shape[0] != B:
                raise ValueError(
                    f"{tensor_name} has a different batch size from img_224: "
                    f"img_224={B}, "
                    f"{tensor_name}={tensor.shape[0]}."
                )

            if tensor.shape[1] != expected_channels[tensor_name]:
                raise ValueError(
                    f"{tensor_name} has an invalid channel count: expected "
                    f"{expected_channels[tensor_name]}, got {tensor.shape[1]}."
                )

            if tensor.shape[-2:] != (H, W):
                raise ValueError(
                    "All four sliding-window inputs must be spatially aligned. "
                    f"img_224={(H, W)}, "
                    f"{tensor_name}={tuple(tensor.shape[-2:])}."
                )

            if tensor.device != img_224.device:
                raise ValueError(
                    f"{tensor_name} is not on the same device as img_224: "
                    f"img_224={img_224.device}, "
                    f"{tensor_name}={tensor.device}."
                )

        # B, _, H, W = img_224.shape

        device = img_224.device
        dtype = img_224.dtype

        gaussian_map = self.gaussian_map.to(device=device, dtype=dtype)

        final_mask_canvas = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        final_zsd_canvas = torch.zeros((B, 64, H, W), device=device, dtype=dtype)

        weight_canvas = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        # --------------------------------------------------
        # Reflection-pad before extracting larger contextual patches when halo
        # inference is enabled.
        # --------------------------------------------------
        img_source = self._pad_for_halo(img_224)

        ma_source = self._pad_for_halo(m_a_224)

        tex_source = self._pad_for_halo(s_tex_224)

        int_source = self._pad_for_halo(s_int_224)

        y_positions = self._get_positions(H)

        x_positions = self._get_positions(W)

        coordinate_list = [(y, x) for y in y_positions for x in x_positions]

        was_training = teacher_b_model.training

        teacher_b_model.eval()

        try:
            for coordinate_chunk in self._iter_chunks(coordinate_list, self.sw_batch_size):
                img_batch = torch.cat(
                    [
                        img_source[:, :, y : y + self.context_size, x : x + self.context_size]
                        for y, x in coordinate_chunk
                    ],
                    dim=0,
                )

                ma_batch = torch.cat(
                    [
                        ma_source[:, :, y : y + self.context_size, x : x + self.context_size]
                        for y, x in coordinate_chunk
                    ],
                    dim=0,
                )

                tex_batch = torch.cat(
                    [
                        tex_source[:, :, y : y + self.context_size, x : x + self.context_size]
                        for y, x in coordinate_chunk
                    ],
                    dim=0,
                )

                int_batch = torch.cat(
                    [
                        int_source[:, :, y : y + self.context_size, x : x + self.context_size]
                        for y, x in coordinate_chunk
                    ],
                    dim=0,
                )

                (_, _, m_final_batch, _, z_sd_batch) = teacher_b_model(
                    img_batch, ma_batch, tex_batch, int_batch
                )

                for patch_index, (y, x) in enumerate(coordinate_chunk):
                    start = patch_index * B

                    end = (patch_index + 1) * B

                    m_final_patch = self._center_crop_last_two_dims(
                        m_final_batch[start:end], crop_size=self.patch_size
                    )

                    z_sd_patch = self._center_crop_last_two_dims(
                        z_sd_batch[start:end], crop_size=self.patch_size
                    )

                    final_mask_canvas[:, :, y : y + self.patch_size, x : x + self.patch_size] += (
                        m_final_patch * gaussian_map
                    )

                    final_zsd_canvas[:, :, y : y + self.patch_size, x : x + self.patch_size] += (
                        z_sd_patch * gaussian_map
                    )

                    weight_canvas[
                        :, :, y : y + self.patch_size, x : x + self.patch_size
                    ] += gaussian_map

        finally:
            if was_training:
                teacher_b_model.train()

        if torch.any(weight_canvas <= 0):
            raise RuntimeError(
                "GaussianSlidingWindowInferer left uncovered pixels; check "
                "patch_size, overlap, and step."
            )

        final_mask = final_mask_canvas / (weight_canvas + 1e-8)

        final_zsd = final_zsd_canvas / (weight_canvas + 1e-8)

        final_mask = torch.clamp(final_mask, min=0.0, max=1.0)

        if not torch.isfinite(final_mask).all():
            raise FloatingPointError("The reconstructed final_mask contains NaN or Inf.")

        if not torch.isfinite(final_zsd).all():
            raise FloatingPointError("The reconstructed final_zsd contains NaN or Inf.")

        return (final_mask, final_zsd)


# SAM is an external dependency and is imported lazily at module load.
try:
    from segment_anything import sam_model_registry
except ImportError:
    sam_model_registry = None


# =====================================================================
# LoRA adapter for the GPPS SAM backbone
# =====================================================================
class LoRA_Linear(nn.Module):
    """Wrap a SAM linear layer with a low-rank adaptation branch."""

    def __init__(self, original_linear, rank=4, alpha=8):
        super().__init__()
        self.original_linear = original_linear
        # Freeze the original linear layer.
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        out_dim, in_dim = original_linear.weight.shape

        # Low-rank matrices A and B.
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

        # A uses Kaiming initialization; zero-initialized B preserves the
        # original network at initialization.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output plus the low-rank branch.
        original_out = self.original_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def inject_lora_to_sam(sam_model, rank=4, alpha=8):
    """Replace every SAM image-encoder QKV projection with a LoRA wrapper."""
    for block in sam_model.image_encoder.blocks:
        original_qkv = block.attn.qkv
        block.attn.qkv = LoRA_Linear(original_qkv, rank=rank, alpha=alpha)
    return sam_model


# =====================================================================
# Polyp-specific GPPS automatic prompt proposals
# =====================================================================
class PolypPromptGeneratorV2(nn.Module):
    """Generate five complete candidate prompt packages for each image.

    Each package contains a dense prompt, positive and negative points, and a
    box.  Top-1 selection is the default; gated Top-2 fusion is enabled only
    when all quality conditions are satisfied.
    """

    def __init__(
        self,
        d_min=4,
        tissue_threshold=0.50,
        distractor_threshold=0.30,
        score_min=0.60,
        fusion_delta=0.08,
        max_candidates=5,
        num_points=8,
    ):
        super().__init__()
        self.d_min = int(d_min)
        self.tissue_threshold = float(tissue_threshold)
        self.distractor_threshold = float(distractor_threshold)
        self.score_min = float(score_min)
        self.fusion_delta = float(fusion_delta)
        self.max_candidates = int(max_candidates)
        self.num_points = int(num_points)
        self.candidate_names = ["objectness", "small", "boundary", "pale_yellow", "conservative"]

    @staticmethod
    def _normalize_np(x, eps=1e-6):
        x = np.asarray(x, dtype=np.float32)
        lo = np.nanpercentile(x, 1)
        hi = np.nanpercentile(x, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)

    def _build_dense_candidates(self, rich):
        objectness = rich["polyp_objectness"]
        tissue = rich["tissue_reliability"]
        distractor = rich["distractor_map"]
        dense_objectness = objectness
        dense_small = torch.clamp(
            0.55 * objectness
            + 0.30 * rich["local_contrast_small"]
            + 0.15 * rich["protrusion_support"],
            0.0,
            1.0,
        )
        dense_boundary = torch.clamp(
            0.55 * objectness
            + 0.25 * rich["objectness_boundary_product"]
            + 0.20 * rich["soft_boundary_support"],
            0.0,
            1.0,
        )
        dense_pale_yellow = torch.clamp(
            0.45 * objectness
            + 0.25 * rich["pale_bright_saliency"]
            + 0.20 * rich["yellow_white_polyp_saliency"]
            + 0.10 * rich["brightness_residual"],
            0.0,
            1.0,
        )
        dense_conservative = torch.clamp(objectness * tissue * (1.0 - distractor), 0.0, 1.0)
        dense_list = [
            dense_objectness,
            dense_small,
            dense_boundary,
            dense_pale_yellow,
            dense_conservative,
        ]
        calibrated = []
        for dense in dense_list:
            dense = torch.clamp(dense, 0.0, 1.0)
            dense = dense * (0.5 + 0.5 * tissue) * (1.0 - 0.5 * distractor)
            calibrated.append(torch.clamp(dense, 0.0, 1.0))
        return torch.stack(calibrated, dim=1)

    @staticmethod
    def _box_area_ratio(box, H, W):
        x1, y1, x2, y2 = box
        return max(0.0, (x2 - x1 + 1.0) * (y2 - y1 + 1.0) / float(max(1, H * W)))

    def _candidate_limits(self, name):
        if name == "small":
            return 0.30, 0.30, 0.45, 0.20
        if name == "boundary":
            return 0.55, 0.30, 0.45, 0.15
        if name == "pale_yellow":
            return 0.50, 0.20, 0.50, 0.15
        if name == "conservative":
            return 0.45, 0.25, 0.50, 0.10
        return 0.60, 0.30, 0.45, 0.15

    def _component_from_dense(self, dense, objectness, tissue, distractor, support, name):
        H, W = dense.shape
        max_area, max_dist, min_tissue, padding_ratio = self._candidate_limits(name)
        thr = max(float(np.nanpercentile(dense, 82)), float(dense.max()) * 0.42, 0.12)
        mask = dense > thr
        if mask.sum() < 6:
            thr = max(float(np.nanpercentile(dense, 75)), float(dense.max()) * 0.35, 0.08)
            mask = dense > thr
        labeled, n_comp = cc_label(mask.astype(np.uint8))
        best = None
        for cid in range(1, n_comp + 1):
            comp = labeled == cid
            area = int(comp.sum())
            if area < 5:
                continue
            ys, xs = np.where(comp)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            area_ratio = area / float(max(1, H * W))
            if area_ratio > max_area:
                continue
            mean_dense = float(dense[comp].mean())
            mean_obj = float(objectness[comp].mean())
            mean_tissue = float(tissue[comp].mean())
            mean_dist = float(distractor[comp].mean())
            mean_support = float(support[comp].mean()) if support is not None else 0.0
            if mean_tissue < min_tissue or mean_dist > max_dist:
                continue
            area_penalty = (
                max(0.0, area_ratio - max_area * 0.75) + max(0.0, 0.0006 - area_ratio) * 3.0
            )
            score = (
                mean_dense + mean_obj + mean_tissue + 0.50 * mean_support - mean_dist - area_penalty
            )
            if best is None or score > best[0]:
                best = (score, comp, (x1, y1, x2, y2), area_ratio)
        if best is None:
            safe = (tissue >= self.tissue_threshold) & (distractor <= self.distractor_threshold)
            score_map = dense * (0.5 + 0.5 * tissue) * (1.0 - 0.5 * distractor)
            score_map = np.where(safe, score_map, -1.0)
            if np.max(score_map) <= 0:
                score_map = dense
            y, x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
            half = 16 if name in {"small", "conservative"} else 24
            x1, x2 = max(0, x - half), min(W - 1, x + half)
            y1, y2 = max(0, y - half), min(H - 1, y + half)
            comp = np.zeros((H, W), dtype=bool)
            comp[y1 : y2 + 1, x1 : x2 + 1] = True
            best = (0.0, comp, (x1, y1, x2, y2), self._box_area_ratio((x1, y1, x2, y2), H, W))
        _, comp, raw_box, _ = best
        x1, y1, x2, y2 = raw_box
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        min_box = 24 if name in {"small", "conservative"} else 32
        pad = int(round(max(bw, bh) * padding_ratio))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(bw + 2 * pad, min_box)
        bh = max(bh + 2 * pad, min_box)
        x1 = int(max(0, round(cx - bw / 2.0)))
        x2 = int(min(W - 1, round(cx + bw / 2.0)))
        y1 = int(max(0, round(cy - bh / 2.0)))
        y2 = int(min(H - 1, round(cy + bh / 2.0)))
        box_mask = np.zeros((H, W), dtype=bool)
        box_mask[y1 : y2 + 1, x1 : x2 + 1] = True
        area_ratio = self._box_area_ratio((x1, y1, x2, y2), H, W)
        distractor_overlap = float(distractor[box_mask].mean()) if box_mask.any() else 1.0
        tissue_mean = float(tissue[box_mask].mean()) if box_mask.any() else 0.0
        objectness_mean = float(objectness[box_mask].mean()) if box_mask.any() else 0.0
        aspect_ratio = max(
            (x2 - x1 + 1) / max(1, (y2 - y1 + 1)), (y2 - y1 + 1) / max(1, (x2 - x1 + 1))
        )
        if (
            area_ratio > max_area
            or distractor_overlap > max_dist
            or tissue_mean < min_tissue
            or objectness_mean < 0.03
            or aspect_ratio > 6.0
        ):
            ys, xs = np.where(comp)
            if len(xs) > 0:
                x_center = int(np.median(xs))
                y_center = int(np.median(ys))
            else:
                y_center, x_center = np.unravel_index(int(np.argmax(dense)), dense.shape)
            half = 16 if name in {"small", "conservative"} else 22
            x1, x2 = max(0, x_center - half), min(W - 1, x_center + half)
            y1, y2 = max(0, y_center - half), min(H - 1, y_center + half)
        return comp.astype(bool), (x1, y1, x2, y2)

    def _sample_positive_points(self, comp, dense, tissue, distractor, box, name):
        H, W = dense.shape
        max_pos = 1 if name in {"small", "conservative"} else 2
        if name in {"objectness", "boundary"} and comp.sum() > 0.03 * H * W:
            max_pos = 3
        elif name == "small" and comp.sum() > 48:
            max_pos = 2
        safe = comp & (tissue >= self.tissue_threshold) & (distractor <= self.distractor_threshold)
        if safe.sum() == 0:
            safe = comp
        dist = distance_transform_edt(safe.astype(np.uint8))
        valid = safe & (dist >= max(1, self.d_min))
        if valid.sum() == 0:
            valid = safe
        score = dense * (0.45 + 0.55 * tissue) * (1.0 - 0.6 * distractor)
        if dist.max() > 0:
            score = score + 0.20 * (dist / (dist.max() + 1e-6))
        score = np.where(valid, score, -1.0)
        points = []
        for _ in range(max_pos):
            if score.max() <= -0.5:
                break
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            points.append((float(x), float(y)))
            yy, xx = np.ogrid[:H, :W]
            suppress_radius = 8 if name != "small" else 5
            score[(yy - y) ** 2 + (xx - x) ** 2 <= suppress_radius**2] = -1.0
        if not points:
            y, x = np.unravel_index(int(np.argmax(dense)), dense.shape)
            points.append((float(x), float(y)))
        return points

    def _point_from_map(self, score_map, forbid_mask):
        arr = np.asarray(score_map, dtype=np.float32).copy()
        arr[forbid_mask] = -1.0
        if arr.max() <= 0:
            return None
        y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return float(x), float(y)

    def _sample_negative_points(
        self, comp, dense, objectness, tissue, distractor, rich_np, box, name
    ):
        H, W = dense.shape
        x1, y1, x2, y2 = box
        forbid = binary_dilation(comp.copy(), iterations=3)
        neg_points = []
        sources = [
            rich_np.get("specular_highlight_map"),
            np.maximum(rich_np.get("dark_lumen_map"), rich_np.get("black_border_map")),
            rich_np.get("instrument_or_white_artifact_map"),
            rich_np.get("mucus_yellow_artifact_map"),
        ]
        for src in sources:
            if src is None:
                continue
            p = self._point_from_map(src * (0.3 + 0.7 * distractor), forbid)
            if p is not None:
                neg_points.append(p)
                px, py = int(round(p[0])), int(round(p[1]))
                yy, xx = np.ogrid[:H, :W]
                forbid[(yy - py) ** 2 + (xx - px) ** 2 <= 8**2] = True
        ring = np.zeros((H, W), dtype=bool)
        margin = 14
        rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
        rx2, ry2 = min(W - 1, x2 + margin), min(H - 1, y2 + margin)
        ring[ry1 : ry2 + 1, rx1 : rx2 + 1] = True
        inner = np.zeros((H, W), dtype=bool)
        inner[y1 : y2 + 1, x1 : x2 + 1] = True
        ring = ring & (~inner)
        non_polyp_high = ring * (0.50 * dense + 0.30 * distractor + 0.20 * (1.0 - tissue))
        p = self._point_from_map(non_polyp_high, forbid)
        if p is not None:
            neg_points.append(p)
        while len(neg_points) < 5:
            p = self._point_from_map(distractor + 0.2 * (1.0 - tissue), forbid)
            if p is None:
                break
            neg_points.append(p)
            px, py = int(round(p[0])), int(round(p[1]))
            yy, xx = np.ogrid[:H, :W]
            forbid[(yy - py) ** 2 + (xx - px) ** 2 <= 10**2] = True
        return neg_points[:5]

    def forward(self, rich, sam_size=1024):
        dense_candidates = self._build_dense_candidates(rich)[:, : self.max_candidates]
        B, K, _, H, W = dense_candidates.shape
        device = dense_candidates.device
        dtype = dense_candidates.dtype
        scale_x = sam_size / float(W)
        scale_y = sam_size / float(H)
        points_out = torch.zeros((B, K, self.num_points, 2), device=device, dtype=dtype)
        labels_out = torch.full((B, K, self.num_points), -1, device=device, dtype=torch.int64)
        boxes_out = torch.zeros((B, K, 4), device=device, dtype=dtype)
        dense_prompts = F.interpolate(
            dense_candidates.reshape(B * K, 1, H, W),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, K, 1, 256, 256)
        rich_cpu = {
            key: value.detach().float().cpu().numpy()[:, 0]
            for key, value in rich.items()
            if isinstance(value, torch.Tensor) and value.ndim == 4
        }
        for b in range(B):
            objectness = rich_cpu["polyp_objectness"][b]
            tissue = rich_cpu["tissue_reliability"][b]
            distractor = rich_cpu["distractor_map"][b]
            support_maps = {
                "objectness": rich_cpu["protrusion_support"][b],
                "small": rich_cpu["local_contrast_small"][b],
                "boundary": rich_cpu["objectness_boundary_product"][b],
                "pale_yellow": rich_cpu["pale_bright_saliency"][b]
                + rich_cpu["yellow_white_polyp_saliency"][b],
                "conservative": rich_cpu["tissue_reliability"][b]
                * (1.0 - rich_cpu["distractor_map"][b]),
            }
            rich_np_b = {k: v[b] for k, v in rich_cpu.items()}
            for k, name in enumerate(self.candidate_names[:K]):
                dense = self._normalize_np(dense_candidates[b, k, 0].detach().float().cpu().numpy())
                comp, box = self._component_from_dense(
                    dense, objectness, tissue, distractor, support_maps.get(name), name
                )
                pos_points = self._sample_positive_points(
                    comp, dense, tissue, distractor, box, name
                )
                neg_points = self._sample_negative_points(
                    comp, dense, objectness, tissue, distractor, rich_np_b, box, name
                )
                all_points = pos_points + neg_points
                all_labels = [1] * len(pos_points) + [0] * len(neg_points)
                for j, (pt, lab) in enumerate(
                    zip(all_points[: self.num_points], all_labels[: self.num_points])
                ):
                    x, y = pt
                    points_out[b, k, j, 0] = float(x) * scale_x
                    points_out[b, k, j, 1] = float(y) * scale_y
                    labels_out[b, k, j] = int(lab)
                x1, y1, x2, y2 = box
                boxes_out[b, k, 0] = float(x1) * scale_x
                boxes_out[b, k, 1] = float(y1) * scale_y
                boxes_out[b, k, 2] = float(x2) * scale_x
                boxes_out[b, k, 3] = float(y2) * scale_y
        return {
            "points": points_out,
            "labels": labels_out,
            "boxes": boxes_out,
            "dense_prompts": dense_prompts,
            "dense_candidates": dense_candidates,
            "candidate_names": self.candidate_names[:K],
        }

    def select_candidate_masks(self, candidate_probs, rich):
        B, K, _, H, W = candidate_probs.shape
        device = candidate_probs.device
        dtype = candidate_probs.dtype
        objectness = rich["polyp_objectness"].to(device=device, dtype=dtype)
        tissue = rich["tissue_reliability"].to(device=device, dtype=dtype)
        distractor = rich["distractor_map"].to(device=device, dtype=dtype)
        boundary_prior = rich["soft_boundary_support"].to(device=device, dtype=dtype)
        eps = 1e-6
        sobel_x = (
            torch.tensor(
                [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
                device=device,
                dtype=dtype,
            )
            / 8.0
        )
        sobel_y = (
            torch.tensor(
                [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
                device=device,
                dtype=dtype,
            )
            / 8.0
        )
        finals = []
        for b in range(B):
            probs = candidate_probs[b]
            bin_masks = (probs > 0.5).float()
            area = bin_masks.mean(dim=(1, 2, 3))
            obj_overlap = (probs * objectness[b : b + 1]).sum(dim=(1, 2, 3)) / probs.sum(
                dim=(1, 2, 3)
            ).clamp_min(eps)
            tissue_mean = (probs * tissue[b : b + 1]).sum(dim=(1, 2, 3)) / probs.sum(
                dim=(1, 2, 3)
            ).clamp_min(eps)
            distractor_overlap = (probs * distractor[b : b + 1]).sum(dim=(1, 2, 3)) / probs.sum(
                dim=(1, 2, 3)
            ).clamp_min(eps)
            prob_edge = torch.clamp(
                torch.sqrt(
                    F.conv2d(probs, sobel_x, padding=1).square()
                    + F.conv2d(probs, sobel_y, padding=1).square()
                    + 1e-8
                )
                * 5.0,
                0.0,
                1.0,
            )
            boundary_align = (prob_edge * boundary_prior[b : b + 1]).sum(
                dim=(1, 2, 3)
            ) / prob_edge.sum(dim=(1, 2, 3)).clamp_min(eps)
            area_penalty = (
                torch.where(area > 0.60, area - 0.60, torch.zeros_like(area))
                + torch.where(area < 0.001, 0.001 - area, torch.zeros_like(area)) * 10.0
            )
            scores = (
                0.35 * obj_overlap
                + 0.25 * tissue_mean
                + 0.20 * boundary_align
                - 0.25 * distractor_overlap
                - 0.30 * area_penalty
            )
            order = torch.argsort(scores, descending=True)
            top1 = int(order[0].item())
            final_prob = probs[top1 : top1 + 1]
            if K >= 2:
                top2 = int(order[1].item())
                score_top1, score_top2 = scores[top1], scores[top2]
                m1, m2 = bin_masks[top1 : top1 + 1], bin_masks[top2 : top2 + 1]
                inter = (m1 * m2).sum()
                union = ((m1 + m2) > 0).float().sum().clamp_min(eps)
                iou12 = inter / union
                independent_top2_ok = bool(
                    obj_overlap[top2] >= 0.55
                    and tissue_mean[top2] >= 0.60
                    and distractor_overlap[top2] <= 0.10
                    and area[top2] <= 0.20
                )
                allow_fusion = bool(
                    score_top1 >= self.score_min
                    and score_top2 >= self.score_min
                    and (score_top1 - score_top2) <= self.fusion_delta
                    and distractor_overlap[top1] <= 0.15
                    and distractor_overlap[top2] <= 0.15
                    and tissue_mean[top1] >= 0.55
                    and tissue_mean[top2] >= 0.55
                    and iou12 < 0.85
                    and (iou12 >= 0.05 or independent_top2_ok)
                )
                if allow_fusion:
                    denom = torch.clamp(score_top1 + score_top2, min=eps)
                    w1 = score_top1 / denom
                    w2 = score_top2 / denom
                    final_prob = w1 * probs[top1 : top1 + 1] + w2 * probs[top2 : top2 + 1]
                    final_prob = final_prob * (0.5 + 0.5 * tissue[b : b + 1])
                    final_prob = final_prob * (0.5 + 0.5 * objectness[b : b + 1])
                    final_prob = safe_probability_map(final_prob)
            finals.append(final_prob)
        return torch.cat(finals, dim=0)


# =====================================================================
# Automatic prior-derived prompt generator for dermoscopic GPPS
# =====================================================================
class PhysicalPromptGenerator(nn.Module):
    """Generate point, box, and dense SAM prompts from 9+3 image priors."""

    def __init__(self):
        super().__init__()
        # Trainable adapter from 12 prior channels to one dense prompt.
        self.dense_adapter = nn.Sequential(
            # Fuse priors across channels.
            nn.Conv2d(12, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # Reduce smoothly to a single-channel mask.
            nn.Conv2d(8, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, s_tex, s_int, sam_size=1024):
        B, _, H, W = s_tex.shape
        device = s_tex.device
        dtype = s_tex.dtype

        # Compute independent horizontal and vertical scale factors.
        scale_x = sam_size / float(W)
        scale_y = sam_size / float(H)

        points_list = []
        boxes_list = []

        # Generate the trainable dense prompt.
        all_priors = torch.cat([s_tex, s_int], dim=1)

        dense_mask = self.dense_adapter(all_priors)  # [B, 1, H, W]

        dense_prompt = F.interpolate(
            dense_mask, size=(256, 256), mode="bilinear", align_corners=False
        )

        # Point and box construction is non-differentiable; detach only their
        # source while retaining gradients through the dense prompt.
        sparse_source = dense_mask[:, 0].detach()

        for i in range(B):
            curr_mask = sparse_source[i]

            # -------------------------------------------------
            # Point prompt at the strongest semantic response.
            # -------------------------------------------------
            max_idx = torch.argmax(curr_mask.reshape(-1))

            y_p = torch.div(max_idx, W, rounding_mode="trunc").float()

            x_p = (max_idx % W).float()

            point_xy = torch.stack([x_p * scale_x, y_p * scale_y])

            points_list.append(point_xy)

            # -------------------------------------------------
            # Box prompt from a 40%-of-maximum relative threshold.
            # -------------------------------------------------
            threshold = curr_mask.max() * 0.4
            coords = torch.nonzero(curr_mask > threshold)

            use_fallback = coords.shape[0] <= 10

            if not use_fallback:
                y1, x1 = torch.min(coords, dim=0)[0].float()
                y2, x2 = torch.max(coords, dim=0)[0].float()

                # Fall back when the response is too narrow to form a box.
                if (x2 - x1).item() < 2 or (y2 - y1).item() < 2:
                    use_fallback = True

            if use_fallback:
                y1 = y_p - 20.0
                x1 = x_p - 20.0
                y2 = y_p + 20.0
                x2 = x_p + 20.0

            # Retain a wider context around the response.
            padding_x = 10.0 * scale_x
            padding_y = 10.0 * scale_y

            x1_scaled = torch.clamp(x1 * scale_x - padding_x, min=0.0, max=float(sam_size))

            y1_scaled = torch.clamp(y1 * scale_y - padding_y, min=0.0, max=float(sam_size))

            x2_scaled = torch.clamp(x2 * scale_x + padding_x, min=0.0, max=float(sam_size))

            y2_scaled = torch.clamp(y2 * scale_y + padding_y, min=0.0, max=float(sam_size))

            box_xyxy = torch.stack([x1_scaled, y1_scaled, x2_scaled, y2_scaled])

            boxes_list.append(box_xyxy)

        # Shapes required by the SAM prompt encoder:
        # points: [B, 1, 2]
        # labels: [B, 1]
        # boxes:  [B, 4]
        points_tensor = torch.stack(points_list, dim=0).unsqueeze(1).to(device=device, dtype=dtype)

        labels_tensor = torch.ones((B, 1), device=device, dtype=torch.int64)

        boxes_tensor = torch.stack(boxes_list, dim=0).to(device=device, dtype=dtype)

        return (points_tensor, labels_tensor), boxes_tensor, dense_prompt


# =====================================================================
# Global prior prompting segmenter (GPPS)
# =====================================================================
class GlobalPriorPromptingSegmenter(nn.Module):
    """Automatically prompt a LoRA-adapted SAM model with image priors."""

    def __init__(
        self,
        model_type="vit_b",
        checkpoint_path="medsam_vit_b.pth",
        lora_rank=4,
        signal_domain="skin",
    ):
        super().__init__()
        if sam_model_registry is None:
            raise RuntimeError(
                "The 'segment_anything' package is required. Install the "
                "dependency before constructing GPPS."
            )

        self.sam_target_size = 1024  # Native SAM input size.

        # Modality-specific prior and prompt generators.
        self.signal_extractor = SignalPriorExtractor(domain=signal_domain)
        self.prompt_generator = PhysicalPromptGenerator()
        self.polyp_prompt_generator = PolypPromptGeneratorV2()

        # The deterministic prior extractor is not trainable.
        for p in self.signal_extractor.parameters():
            p.requires_grad = False

        # Construct the external SAM/MedSAM backbone from the supplied weights.
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        # Historical pixel-normalization override (disabled to preserve the
        # trained architecture and checkpoint behavior).
        # The original pixel_mean and pixel_std buffers remain unchanged.

        # Freeze the backbone before adding the trainable adapters.
        for param in self.sam.parameters():
            param.requires_grad = False

        # Train LoRA adapters in the image-encoder attention blocks.
        self.sam = inject_lora_to_sam(self.sam, rank=lora_rank)

        # Fine-tune the mask decoder for lesion masks.
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

        # The custom generator supplies differentiable dense prompts; the SAM
        # prompt encoder itself remains frozen.

    def forward(self, x_full):
        """Predict a full-resolution coarse probability mask.

        The skin domain uses the 9+3 prior prompt generator.  The polyp domain
        uses five candidate-specific prompt packages followed by mask selection.
        """
        B, C, H, W = x_full.shape
        if C != 3:
            raise ValueError(
                "GlobalPriorPromptingSegmenter expects RGB input, " f"but received C={C}."
            )
        x_rgb = self.signal_extractor.to_physical_rgb(x_full)

        if self.signal_extractor.domain == "polyp":
            with torch.no_grad():
                rich_prior = self.signal_extractor.extract_polyp_rich(x_full)
            prompt_pack = self.polyp_prompt_generator(
                rich=rich_prior, sam_size=self.sam_target_size
            )
        else:
            with torch.no_grad():
                s_tex, s_int = self.signal_extractor(x_full)
            sparse_points_tuple, sparse_boxes, dense_mask = self.prompt_generator(
                s_tex=s_tex, s_int=s_int, sam_size=self.sam_target_size
            )
            prompt_pack = None

        x_1024 = F.interpolate(
            x_rgb,
            size=(self.sam_target_size, self.sam_target_size),
            mode="bilinear",
            align_corners=False,
        )
        x_min = x_1024.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x_1024.amax(dim=(1, 2, 3), keepdim=True)
        x_1024 = (x_1024 - x_min) / torch.clamp(x_max - x_min, min=1e-8)
        image_embeddings = self.sam.image_encoder(x_1024)

        if self.signal_extractor.domain == "polyp":
            K = prompt_pack["points"].shape[1]
            candidate_probs = []
            for k in range(K):
                low_res_masks_list = []
                for i in range(B):
                    curr_img_emb = image_embeddings[i : i + 1]
                    curr_points = (
                        prompt_pack["points"][i : i + 1, k],
                        prompt_pack["labels"][i : i + 1, k],
                    )
                    curr_boxes = prompt_pack["boxes"][i : i + 1, k]
                    curr_dense_mask = prompt_pack["dense_prompts"][i : i + 1, k]
                    sparse_emb, dense_emb = self.sam.prompt_encoder(
                        points=curr_points, boxes=curr_boxes, masks=curr_dense_mask
                    )
                    low_res_logits, _ = self.sam.mask_decoder(
                        image_embeddings=curr_img_emb,
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    low_res_masks_list.append(low_res_logits)
                low_res_logits_k = torch.cat(low_res_masks_list, dim=0)
                prob_k = safe_probability_map(torch.sigmoid(low_res_logits_k))
                prob_k = F.interpolate(prob_k, size=(H, W), mode="bilinear", align_corners=False)
                candidate_probs.append(prob_k)
            candidate_probs = torch.stack(candidate_probs, dim=1)
            M_A_full = self.polyp_prompt_generator.select_candidate_masks(
                candidate_probs, rich_prior
            )
            return safe_probability_map(M_A_full)

        low_res_masks_list = []
        for i in range(B):
            curr_img_emb = image_embeddings[i : i + 1]
            curr_points = (sparse_points_tuple[0][i : i + 1], sparse_points_tuple[1][i : i + 1])
            curr_boxes = sparse_boxes[i : i + 1]
            curr_dense_mask = dense_mask[i : i + 1]
            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=curr_points, boxes=curr_boxes, masks=curr_dense_mask
            )
            low_res_logits, _ = self.sam.mask_decoder(
                image_embeddings=curr_img_emb,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            low_res_masks_list.append(low_res_logits)
        low_res_logits = torch.cat(low_res_masks_list, dim=0)
        low_res_probs = safe_probability_map(torch.sigmoid(low_res_logits))
        M_A_full = F.interpolate(low_res_probs, size=(H, W), mode="bilinear", align_corners=False)
        return safe_probability_map(M_A_full)

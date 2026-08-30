import numpy as np
import pytest

from kglr_net.metrics import compute_segmentation_metrics, hd95


def test_identical_masks_are_perfect() -> None:
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 5:11] = 1.0
    metrics = compute_segmentation_metrics(mask, mask)
    assert metrics.dice == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.hd95 == pytest.approx(0.0)


def test_one_empty_mask_uses_image_diagonal() -> None:
    empty = np.zeros((3, 4), dtype=np.float32)
    nonempty = empty.copy()
    nonempty[1, 1] = 1.0
    assert hd95(empty, nonempty) == pytest.approx(5.0)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compute_segmentation_metrics(np.zeros((4, 4)), np.zeros((5, 5)))

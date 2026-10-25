"""
Unit tests for src/preprocessing.py
"""

import pytest
import numpy as np
import cv2
from src.preprocessing import (
    to_grayscale,
    apply_filter,
    apply_threshold,
    apply_morphology,
    preprocess_image,
)


def test_to_grayscale_valid_inputs():
    # 3-channel BGR
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[:, :] = (50, 100, 150)
    gray = to_grayscale(bgr)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    # Already grayscale
    gray_in = np.ones((50, 50), dtype=np.uint8) * 120
    gray_out = to_grayscale(gray_in)
    assert gray_out.shape == (50, 50)
    assert np.array_equal(gray_in, gray_out)

    # 4-channel BGRA
    bgra = np.zeros((40, 40, 4), dtype=np.uint8)
    gray_bgra = to_grayscale(bgra)
    assert gray_bgra.shape == (40, 40)


def test_to_grayscale_invalid_inputs():
    with pytest.raises(ValueError):
        to_grayscale(None)
    with pytest.raises(TypeError):
        to_grayscale("not an image")


def test_apply_filter():
    gray = np.random.randint(0, 255, (80, 80), dtype=np.uint8)

    # Gaussian
    gauss = apply_filter(gray, method="Gaussian", kernel_size=5, sigma=1.2)
    assert gauss.shape == gray.shape

    # Even kernel size should be auto-adjusted to odd
    gauss_even = apply_filter(gray, method="Gaussian", kernel_size=4)
    assert gauss_even.shape == gray.shape

    # Median
    med = apply_filter(gray, method="Median", kernel_size=3)
    assert med.shape == gray.shape

    # None
    no_filter = apply_filter(gray, method="None")
    assert np.array_equal(gray, no_filter)


def test_apply_threshold_manual_and_otsu():
    # Synthetic gradient image
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:, 50:] = 200

    # Manual
    mask, info = apply_threshold(gray, method="Manual", thresh_val=100, invert=False)
    assert mask.shape == (100, 100)
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[0, 10] == 0
    assert mask[0, 60] == 255
    assert info["effective_threshold"] == 100

    # Invert
    mask_inv, info_inv = apply_threshold(gray, method="Manual", thresh_val=100, invert=True)
    assert mask_inv[0, 10] == 255
    assert mask_inv[0, 60] == 0
    assert info_inv["inverted"] is True

    # Otsu on bimodal distribution
    bimodal = np.zeros((100, 100), dtype=np.uint8)
    bimodal[:50, :] = 40
    bimodal[50:, :] = 190
    mask_otsu, info_otsu = apply_threshold(bimodal, method="Otsu")
    assert mask_otsu.shape == (100, 100)
    assert "effective_threshold" in info_otsu
    assert info_otsu["effective_threshold"] >= 40.0


def test_apply_threshold_adaptive():
    gray = np.random.randint(100, 200, (60, 60), dtype=np.uint8)
    mask_adapt, info = apply_threshold(
        gray, method="Adaptive", adaptive_method="Gaussian", block_size=11, c_val=2
    )
    assert mask_adapt.shape == (60, 60)
    assert set(np.unique(mask_adapt)).issubset({0, 255})


def test_apply_morphology():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)

    eroded = apply_morphology(mask, operation="Erosion", kernel_size=5, iterations=1)
    dilated = apply_morphology(mask, operation="Dilation", kernel_size=5, iterations=1)
    opening = apply_morphology(mask, operation="Opening", kernel_size=3, iterations=1)
    closing = apply_morphology(mask, operation="Closing", kernel_size=3, iterations=1)

    assert np.count_nonzero(eroded) < np.count_nonzero(mask)
    assert np.count_nonzero(dilated) > np.count_nonzero(mask)
    assert opening.shape == mask.shape
    assert closing.shape == mask.shape


def test_preprocess_pipeline():
    img = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.circle(img, (80, 60), 25, (30, 30, 30), -1)

    gray, clean_mask, info = preprocess_image(
        img,
        filter_method="Gaussian",
        filter_kernel=5,
        thresh_method="Otsu",
        invert_polarity=True,  # Dark circle on light background -> invert to get white foreground
        morph_operation="Opening",
    )

    assert gray.shape == (120, 160)
    assert clean_mask.shape == (120, 160)
    assert np.count_nonzero(clean_mask) > 0
    assert info["dimensions"]["width"] == 160
    assert info["dimensions"]["height"] == 120

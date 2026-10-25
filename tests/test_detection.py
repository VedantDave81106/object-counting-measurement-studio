"""
Unit tests for src/detection.py
"""

import pytest
import numpy as np
import cv2
from src.detection import detect_contours, segment_watershed
from src.sample_generator import generate_hardware_washers_image


def test_detect_contours_known_count():
    # Create mask with 4 separated white squares (40x40 each = 1600 px^2 area)
    mask = np.zeros((300, 300), dtype=np.uint8)
    coords = [(30, 30), (120, 30), (30, 150), (150, 150)]
    for x, y in coords:
        cv2.rectangle(mask, (x, y), (x + 40, y + 40), 255, -1)

    contours, stats = detect_contours(mask, min_area=100, max_area=10_000)
    assert len(contours) == 4
    assert stats["filtered_count"] == 4
    assert stats["raw_count"] == 4
    assert stats["rejected_too_small"] == 0


def test_detect_contours_area_filtering():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Add 1 tiny noise dot (3x3 = 9 px^2)
    cv2.rectangle(mask, (10, 10), (13, 13), 255, -1)
    # Add 1 valid object (30x30 = 900 px^2)
    cv2.rectangle(mask, (50, 50), (80, 80), 255, -1)

    # Filtering with min_area = 50 should reject the 9 px^2 speckle
    contours, stats = detect_contours(mask, min_area=50)
    assert len(contours) == 1
    assert stats["rejected_too_small"] == 1


def test_detect_contours_empty_mask():
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    contours, stats = detect_contours(empty_mask)
    assert len(contours) == 0
    assert stats["filtered_count"] == 0

    none_contours, none_stats = detect_contours(None)
    assert len(none_contours) == 0


def test_holes_not_counted_as_objects():
    """
    Critical IPCV concept: In RETR_EXTERNAL mode, an object with an internal hole
    (like a washer or donut) must be detected as 1 object, NOT 2 objects.
    """
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Outer ring radius 40, inner hole radius 18
    cv2.circle(mask, (100, 100), 40, 255, -1)
    cv2.circle(mask, (100, 100), 18, 0, -1)  # Cut hole

    contours, stats = detect_contours(mask, min_area=50)
    assert len(contours) == 1  # Crucial: Outer contour only, hole not counted!


def test_segment_watershed():
    # Two kissing disks
    img = np.ones((150, 300, 3), dtype=np.uint8) * 255
    mask = np.zeros((150, 300), dtype=np.uint8)

    cv2.circle(mask, (100, 75), 35, 255, -1)
    cv2.circle(mask, (160, 75), 35, 255, -1)

    # Standard contours without watershed sees this as 1 fused contour
    raw_cnts, _ = detect_contours(mask, min_area=50)
    assert len(raw_cnts) == 1

    # Watershed segmentation splits them into 2 objects
    ws_cnts, markers, stats = segment_watershed(img, mask, min_distance_ratio=0.6, min_area=50)
    assert len(ws_cnts) == 2
    assert stats["filtered_count"] == 2

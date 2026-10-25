"""
Unit tests for src/measurement.py and src/export.py
"""

import json
import math
import numpy as np
import cv2
from src.measurement import measure_contour, measure_all_objects, compute_summary_statistics
from src.export import measurements_to_csv, generate_json_report, encode_image_png


def test_measure_circle_geometry():
    # Draw circle with radius = 30
    img = np.zeros((100, 100), dtype=np.uint8)
    r = 30
    cv2.circle(img, (50, 50), r, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1

    m = measure_contour(contours[0], object_id=1)
    expected_area = math.pi * r * r

    # Discretized pixel contour area should be within 5% of theoretical area
    assert abs(m["area_px"] - expected_area) / expected_area < 0.05
    # Circularity for circle should be close to 1.0 (>= 0.85 on discrete grid)
    assert m["circularity"] >= 0.85
    # Aspect ratio should be close to 1.0
    assert abs(m["aspect_ratio"] - 1.0) < 0.1
    # Equiv diameter close to 2*r
    assert abs(m["equiv_diameter_px"] - 2 * r) < 3.0


def test_measure_square_geometry():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (70, 70), 255, -1)  # 50x50 = 2500 px^2
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1

    m = measure_contour(contours[0], object_id=1)
    # Area for 50x50 on grid in OpenCV contourArea
    assert abs(m["area_px"] - 2400) < 150
    assert m["bbox_width_px"] in (49, 50, 51)
    assert m["bbox_height_px"] in (49, 50, 51)
    assert abs(m["aspect_ratio"] - 1.0) < 0.05
    # Theoretical circularity of square is pi/4 = 0.785
    assert 0.70 <= m["circularity"] <= 0.85


def test_real_world_calibration():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (70, 70), 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calibrate: 10 pixels = 1 mm -> pixels_per_unit = 10.0
    m = measure_contour(contours[0], object_id=1, pixels_per_unit=10.0, unit_name="mm")
    assert m["is_calibrated"] is True
    assert m["unit"] == "mm"
    # Width ~50 px -> ~5.0 mm
    assert abs(m["width_physical"] - 5.0) < 0.3
    # Area ~2400-2500 px^2 -> ~24-25 mm^2
    assert abs(m["area_physical"] - 25.0) < 2.0


def test_compute_summary_statistics():
    dummy_records = [
        {"object_id": 1, "area_px": 100.0, "perimeter_px": 40.0, "bbox_width_px": 10, "bbox_height_px": 10, "circularity": 0.8, "is_calibrated": False},
        {"object_id": 2, "area_px": 200.0, "perimeter_px": 60.0, "bbox_width_px": 20, "bbox_height_px": 10, "circularity": 0.6, "is_calibrated": False},
    ]

    summary = compute_summary_statistics(dummy_records)
    assert summary["total_count"] == 2
    assert summary["avg_area"] == 150.0
    assert summary["min_area"] == 100.0
    assert summary["max_area"] == 200.0
    assert summary["avg_perimeter"] == 50.0

    # Test empty summary
    empty_summary = compute_summary_statistics([])
    assert empty_summary["total_count"] == 0
    assert empty_summary["avg_area"] == 0.0


def test_export_csv_and_json():
    dummy_records = [
        {
            "object_id": 1,
            "area_px": 1500.0,
            "perimeter_px": 160.0,
            "bbox_x": 10,
            "bbox_y": 10,
            "bbox_width_px": 40,
            "bbox_height_px": 40,
            "aspect_ratio": 1.0,
            "equiv_diameter_px": 43.7,
            "circularity": 0.74,
            "centroid_x": 30.0,
            "centroid_y": 30.0,
            "is_calibrated": False,
            "unit": "px",
        }
    ]
    summary = compute_summary_statistics(dummy_records)

    # 1. CSV
    csv_str = measurements_to_csv(dummy_records)
    assert "object_id" in csv_str
    assert "area_px" in csv_str
    assert "1500" in csv_str

    # 2. JSON
    json_str = generate_json_report(
        dummy_records,
        summary,
        image_metadata={"width": 640, "height": 480},
        preprocessing_params={"filter": "Gaussian"},
        detection_params={"min_area": 100},
        calibration_info={"calibrated": False},
    )
    parsed = json.loads(json_str)
    assert parsed["summary_statistics"]["total_count"] == 1
    assert len(parsed["detected_objects"]) == 1

    # 3. Image PNG encoding
    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    png_bytes = encode_image_png(dummy_img)
    assert len(png_bytes) > 0
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

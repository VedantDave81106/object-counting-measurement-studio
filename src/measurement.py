"""
Object Measurement Module for Object Counting and Measurement Studio.

Computes geometric features for detected contours:
- Area (cv2.contourArea)
- Perimeter / Arc Length (cv2.arcLength)
- Bounding Box Dimensions & Aspect Ratio (cv2.boundingRect)
- Equivalent Circular Diameter (2 * sqrt(Area / pi))
- Circularity / Form Factor (4 * pi * Area / Perimeter^2)
- Centroid (Moments M10/M00, M01/M00)
- Optional Real-World Calibration (conversion to mm, cm, inches, etc.)
"""

import math
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


def measure_contour(
    contour: np.ndarray,
    object_id: int,
    pixels_per_unit: float = 1.0,
    unit_name: str = "px",
) -> Dict[str, Any]:
    """
    Compute rigorous 2D geometric measurements for a single contour.

    Args:
        contour: Contour points array (Nx1x2 or Nx2).
        object_id: Unique integer identifier for the object.
        pixels_per_unit: Calibration ratio (number of pixels per physical unit, e.g., px/mm).
        unit_name: Physical measurement unit name ('px', 'mm', 'cm', 'in').

    Returns:
        Dictionary containing image-space and calibrated physical measurements.
    """
    # 1. Area & Perimeter
    area_px = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, closed=True))

    # 2. Bounding Box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w / h) if h > 0 else 0.0

    # 3. Equivalent Circular Diameter
    # D = 2 * sqrt(Area / pi)
    equiv_diameter_px = 2.0 * math.sqrt(area_px / math.pi) if area_px > 0 else 0.0

    # 4. Circularity / Isoperimetric Quotient
    # C = (4 * pi * Area) / (Perimeter^2)
    # Circle = 1.0, Square = ~0.785, Star / Spiky < 0.5
    if perimeter_px > 0:
        raw_circularity = (4.0 * math.pi * area_px) / (perimeter_px * perimeter_px)
        # Numerical discretization on pixel grids can slightly exceed 1.0 for tiny shapes
        circularity = min(1.0, max(0.0, float(raw_circularity)))
    else:
        circularity = 0.0

    # 5. Centroid via Image Moments
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)

    # 6. Physical Calibrated Units
    is_calibrated = (pixels_per_unit > 0) and (pixels_per_unit != 1.0 or unit_name != "px")
    scale_factor = float(pixels_per_unit) if pixels_per_unit > 0 else 1.0
    area_scale = scale_factor * scale_factor

    area_calibrated = area_px / area_scale
    perimeter_calibrated = perimeter_px / scale_factor
    width_calibrated = float(w) / scale_factor
    height_calibrated = float(h) / scale_factor
    equiv_diameter_calibrated = equiv_diameter_px / scale_factor

    return {
        "object_id": int(object_id),
        "area_px": round(area_px, 2),
        "perimeter_px": round(perimeter_px, 2),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_width_px": int(w),
        "bbox_height_px": int(h),
        "aspect_ratio": round(aspect_ratio, 3),
        "equiv_diameter_px": round(equiv_diameter_px, 2),
        "circularity": round(circularity, 3),
        "centroid_x": round(cx, 1),
        "centroid_y": round(cy, 1),
        # Physical calibrated fields
        "is_calibrated": is_calibrated,
        "unit": unit_name,
        "area_physical": round(area_calibrated, 3),
        "perimeter_physical": round(perimeter_calibrated, 3),
        "width_physical": round(width_calibrated, 3),
        "height_physical": round(height_calibrated, 3),
        "equiv_diameter_physical": round(equiv_diameter_calibrated, 3),
    }


def measure_all_objects(
    contours: List[np.ndarray],
    pixels_per_unit: float = 1.0,
    unit_name: str = "px",
) -> List[Dict[str, Any]]:
    """
    Measure all detected contours and return a list of measurement records.
    """
    results: List[Dict[str, Any]] = []
    for idx, cnt in enumerate(contours, start=1):
        record = measure_contour(
            cnt,
            object_id=idx,
            pixels_per_unit=pixels_per_unit,
            unit_name=unit_name,
        )
        results.append(record)
    return results


def compute_summary_statistics(
    measurements: List[Dict[str, Any]],
    unit_name: str = "px",
) -> Dict[str, Any]:
    """
    Compute aggregate summary metrics across all detected objects.
    """
    count = len(measurements)
    if count == 0:
        return {
            "total_count": 0,
            "avg_area": 0.0,
            "min_area": 0.0,
            "max_area": 0.0,
            "avg_perimeter": 0.0,
            "avg_width": 0.0,
            "avg_height": 0.0,
            "avg_circularity": 0.0,
            "unit": unit_name,
        }

    is_calibrated = measurements[0].get("is_calibrated", False)
    area_key = "area_physical" if is_calibrated else "area_px"
    perim_key = "perimeter_physical" if is_calibrated else "perimeter_px"
    width_key = "width_physical" if is_calibrated else "bbox_width_px"
    height_key = "height_physical" if is_calibrated else "bbox_height_px"

    areas = [m[area_key] for m in measurements]
    perimeters = [m[perim_key] for m in measurements]
    widths = [m[width_key] for m in measurements]
    heights = [m[height_key] for m in measurements]
    circularities = [m["circularity"] for m in measurements]

    return {
        "total_count": count,
        "avg_area": round(float(np.mean(areas)), 2),
        "min_area": round(float(np.min(areas)), 2),
        "max_area": round(float(np.max(areas)), 2),
        "avg_perimeter": round(float(np.mean(perimeters)), 2),
        "avg_width": round(float(np.mean(widths)), 2),
        "avg_height": round(float(np.mean(heights)), 2),
        "avg_circularity": round(float(np.mean(circularities)), 3),
        "is_calibrated": is_calibrated,
        "unit": unit_name,
    }

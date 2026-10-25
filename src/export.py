"""
Export Module for Object Counting and Measurement Studio.

Provides real file generators and serializers:
- PNG byte buffer encoding for annotated images and binary masks
- CSV serialization for measurement data tables (Pandas)
- Comprehensive JSON analysis report containing parameters, dimensions, and measurements
"""

import json
from typing import List, Dict, Any
import cv2
import numpy as np
import pandas as pd


def encode_image_png(image: np.ndarray) -> bytes:
    """
    Encode an OpenCV image (BGR or Grayscale) into PNG bytes suitable for file download.

    Args:
        image: NumPy image array.

    Returns:
        PNG encoded bytes.
    """
    if image is None or image.size == 0:
        return b""

    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format")
    return buffer.tobytes()


def measurements_to_dataframe(measurements: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert measurement records to a clean Pandas DataFrame for display and CSV export.
    """
    if not measurements:
        return pd.DataFrame()

    df = pd.DataFrame(measurements)

    # Reorder columns logically
    primary_cols = [
        "object_id",
        "area_px",
        "perimeter_px",
        "bbox_width_px",
        "bbox_height_px",
        "aspect_ratio",
        "equiv_diameter_px",
        "circularity",
        "centroid_x",
        "centroid_y",
    ]

    is_calibrated = measurements[0].get("is_calibrated", False)
    if is_calibrated:
        unit = measurements[0].get("unit", "unit")
        calib_cols = [
            "unit",
            "area_physical",
            "perimeter_physical",
            "width_physical",
            "height_physical",
            "equiv_diameter_physical",
        ]
        ordered_cols = (
            ["object_id"]
            + calib_cols
            + [c for c in primary_cols if c not in ["object_id"] and c in df.columns]
        )
    else:
        ordered_cols = [c for c in primary_cols if c in df.columns]

    # Add any extra remaining columns
    remaining = [c for c in df.columns if c not in ordered_cols and c != "is_calibrated"]
    final_cols = [c for c in (ordered_cols + remaining) if c in df.columns]

    return df[final_cols]


def measurements_to_csv(measurements: List[Dict[str, Any]]) -> str:
    """
    Serialize measurements into CSV string.
    """
    df = measurements_to_dataframe(measurements)
    return df.to_csv(index=False)


def generate_json_report(
    measurements: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    image_metadata: Dict[str, Any],
    preprocessing_params: Dict[str, Any],
    detection_params: Dict[str, Any],
    calibration_info: Dict[str, Any],
) -> str:
    """
    Generate comprehensive JSON analysis report document.
    """
    report = {
        "project": "Object Counting and Measurement Studio",
        "course": "Image Processing and Computer Vision (IPCV) Laboratory",
        "image_metadata": image_metadata,
        "calibration": calibration_info,
        "preprocessing_parameters": preprocessing_params,
        "detection_parameters": detection_params,
        "summary_statistics": summary_stats,
        "detected_objects": measurements,
    }

    return json.dumps(report, indent=4)

"""
Object Detection Module for Object Counting and Measurement Studio.

Performs classical contour extraction, area filtering, boundary detection,
and optional distance-transform watershed segmentation for touching objects.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def detect_contours(
    binary_mask: np.ndarray,
    min_area: float = 50.0,
    max_area: float = 1_000_000.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extract external contours from a binary mask and filter them by area.

    Uses cv2.RETR_EXTERNAL to ensure internal holes in objects (e.g., washers,
    donuts) are not counted as separate objects.

    Args:
        binary_mask: Single-channel binary image (uint8, 0 and 255).
        min_area: Minimum contour area in pixels^2 to reject noise.
        max_area: Maximum contour area in pixels^2 to reject border/background artifacts.

    Returns:
        (valid_contours, detection_stats)
    """
    if binary_mask is None or binary_mask.size == 0:
        return [], {"raw_count": 0, "filtered_count": 0, "rejected_count": 0}

    # Ensure mask is 8-bit single channel
    if binary_mask.ndim == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    # cv2.findContours returns (contours, hierarchy)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours: List[np.ndarray] = []
    rejected_too_small = 0
    rejected_too_large = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            rejected_too_small += 1
            continue
        if max_area is not None and area > max_area:
            rejected_too_large += 1
            continue
        valid_contours.append(cnt)

    # Sort contours by spatial position (top-to-bottom, left-to-right) for consistent numbering
    if valid_contours:
        valid_contours.sort(
            key=lambda c: (cv2.boundingRect(c)[1] // 30, cv2.boundingRect(c)[0])
        )

    stats = {
        "raw_count": len(contours),
        "filtered_count": len(valid_contours),
        "rejected_too_small": rejected_too_small,
        "rejected_too_large": rejected_too_large,
    }

    return valid_contours, stats


def segment_watershed(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    min_distance_ratio: float = 0.5,
    min_area: float = 50.0,
    max_area: float = 1_000_000.0,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Optional Distance-Transform and Marker-Controlled Watershed segmentation.
    Useful for separating touching or slightly overlapping convex objects (e.g., coins, pills, cells).

    Steps:
    1. Distance Transform on binary mask (Euclidean L2 distance to nearest background).
    2. Threshold distance map to extract sure foreground marker peaks.
    3. Dilate binary mask to determine sure background.
    4. Compute unknown boundary region (sure background - sure foreground).
    5. Label connected components on sure foreground as markers.
    6. Apply cv2.watershed to propagate boundaries.
    7. Extract external contours for each labeled basin.

    Args:
        image_bgr: 3-channel color image (watershed requires 3-channel).
        binary_mask: Single-channel binary mask.
        min_distance_ratio: Ratio of max distance to treat as foreground marker (0.1 - 0.9).
        min_area: Minimum area filter.
        max_area: Maximum area filter.

    Returns:
        (contours, watershed_markers_image, stats)
    """
    if binary_mask is None or binary_mask.size == 0 or cv2.countNonZero(binary_mask) == 0:
        return [], np.zeros_like(binary_mask), {"raw_count": 0, "filtered_count": 0}

    # Ensure 3-channel BGR image
    if image_bgr.ndim == 2:
        image_bgr_3c = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr_3c = image_bgr.copy()

    # 1. Morphological opening to eliminate any stray 1-pixel bridges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 3. Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()

    if max_dist <= 0:
        return [], np.zeros_like(binary_mask), {"raw_count": 0, "filtered_count": 0}

    # 4. Sure foreground via distance threshold
    _, sure_fg = cv2.threshold(
        dist_transform, float(min_distance_ratio) * max_dist, 255, cv2.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    # 5. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labelling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Mark the region of unknown with 0
    markers[unknown == 255] = 0

    # 7. Apply watershed
    markers_ws = markers.copy()
    cv2.watershed(image_bgr_3c, markers_ws)

    # 8. Reconstruct separated contours from segmented regions
    valid_contours: List[np.ndarray] = []
    # Markers range from 2 to num_markers for each object
    for label in range(2, num_markers + 1):
        obj_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        obj_mask[(markers_ws == label) & (binary_mask > 0)] = 255
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                valid_contours.append(c)

    if valid_contours:
        valid_contours.sort(
            key=lambda c: (cv2.boundingRect(c)[1] // 30, cv2.boundingRect(c)[0])
        )

    stats = {
        "raw_count": num_markers - 1,
        "filtered_count": len(valid_contours),
        "method": "Marker-Controlled Watershed",
    }

    return valid_contours, markers_ws, stats

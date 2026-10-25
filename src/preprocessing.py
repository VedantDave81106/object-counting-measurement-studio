"""
Image Preprocessing Module for Object Counting and Measurement Studio.

Provides classical IPCV preprocessing operations:
- Color space conversion (BGR/RGB to Grayscale)
- Smoothing & Filtering (Gaussian, Median, Normalized Box)
- Thresholding (Manual, Otsu's Global Automatic, Adaptive Mean/Gaussian)
- Binary Inversion (Foreground Polarity Selection)
- Morphological Operations (Erosion, Dilation, Opening, Closing)
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an input image (BGR, RGB, or single-channel) to grayscale.

    Args:
        image: Input NumPy image array.

    Returns:
        Single-channel 8-bit grayscale image.
    """
    if image is None:
        raise ValueError("Input image cannot be None")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy.ndarray")

    if image.ndim == 2:
        # Already single channel
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image.copy()

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy()
        elif channels == 3:
            # Assume BGR standard format for OpenCV
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

    raise ValueError(f"Invalid image array dimension: {image.ndim}")


def apply_filter(
    image_gray: np.ndarray,
    method: str = "Gaussian",
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply smoothing or noise reduction filter to a grayscale image.

    Args:
        image_gray: Grayscale 8-bit image.
        method: Filter type - 'None', 'Gaussian', 'Median', or 'Box'.
        kernel_size: Size of kernel (must be a positive odd integer).
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        Filtered grayscale image.
    """
    if image_gray is None or image_gray.size == 0:
        raise ValueError("Grayscale image is empty or None")

    if method == "None" or kernel_size <= 1:
        return image_gray.copy()

    # Ensure kernel_size is positive and odd
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    if method == "Gaussian":
        return cv2.GaussianBlur(image_gray, (k, k), sigmaX=float(sigma))
    elif method == "Median":
        return cv2.medianBlur(image_gray, k)
    elif method == "Box":
        return cv2.blur(image_gray, (k, k))
    else:
        raise ValueError(f"Unknown filter method: {method}")


def apply_threshold(
    image_gray: np.ndarray,
    method: str = "Otsu",
    thresh_val: int = 127,
    invert: bool = False,
    adaptive_method: str = "Gaussian",
    block_size: int = 11,
    c_val: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform image thresholding to segment foreground objects from background.

    Args:
        image_gray: Grayscale 8-bit image.
        method: Thresholding technique - 'Manual', 'Otsu', or 'Adaptive'.
        thresh_val: Manual threshold cut-off (0-255).
        invert: Invert binary polarity (True: foreground is darker than background).
        adaptive_method: 'Mean' (ADAPTIVE_THRESH_MEAN_C) or 'Gaussian' (ADAPTIVE_THRESH_GAUSSIAN_C).
        block_size: Neighborhood size for adaptive thresholding (odd integer >= 3).
        c_val: Constant subtracted from the mean or weighted mean in adaptive thresholding.

    Returns:
        Tuple of (binary_mask, info_dict) where binary_mask has values 0 (background) and 255 (foreground).
    """
    if image_gray is None or image_gray.size == 0:
        raise ValueError("Grayscale image is empty or None")

    base_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    info: Dict[str, Any] = {
        "method": method,
        "inverted": invert,
    }

    if method == "Manual":
        t_val = int(np.clip(thresh_val, 0, 255))
        _, binary = cv2.threshold(image_gray, t_val, 255, base_type)
        info["effective_threshold"] = t_val

    elif method == "Otsu":
        # Otsu computes optimal threshold minimizing intra-class variance
        flags = base_type | cv2.THRESH_OTSU
        computed_thresh, binary = cv2.threshold(image_gray, 0, 255, flags)
        info["effective_threshold"] = float(computed_thresh)

    elif method == "Adaptive":
        # Ensure block_size is odd and >= 3
        b_size = max(3, int(block_size))
        if b_size % 2 == 0:
            b_size += 1

        adapt_flag = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if adaptive_method == "Gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        binary = cv2.adaptiveThreshold(
            image_gray, 255, adapt_flag, base_type, b_size, int(c_val)
        )
        info["effective_threshold"] = f"Adaptive ({adaptive_method}, block={b_size}, C={c_val})"

    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    return binary, info


def apply_morphology(
    binary_mask: np.ndarray,
    operation: str = "None",
    kernel_size: int = 3,
    kernel_shape: str = "Rectangle",
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological operations to clean binary mask (remove noise, bridge gaps, separate contacts).

    Args:
        binary_mask: Binary 8-bit image (0 and 255).
        operation: 'None', 'Erosion', 'Dilation', 'Opening', 'Closing'.
        kernel_size: Size of structuring element (odd integer >= 1).
        kernel_shape: 'Rectangle', 'Ellipse', or 'Cross'.
        iterations: Number of times the morphological operation is applied.

    Returns:
        Processed binary mask.
    """
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("Binary mask is empty or None")

    if operation == "None" or kernel_size < 1 or iterations < 1:
        return binary_mask.copy()

    k_size = max(1, int(kernel_size))
    shape_map = {
        "Rectangle": cv2.MORPH_RECT,
        "Ellipse": cv2.MORPH_ELLIPSE,
        "Cross": cv2.MORPH_CROSS,
    }
    morph_shape = shape_map.get(kernel_shape, cv2.MORPH_RECT)
    element = cv2.getStructuringElement(morph_shape, (k_size, k_size))

    if operation == "Erosion":
        return cv2.erode(binary_mask, element, iterations=int(iterations))
    elif operation == "Dilation":
        return cv2.dilate(binary_mask, element, iterations=int(iterations))
    elif operation == "Opening":
        # Opening = Erosion followed by Dilation (removes small background noise)
        return cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, element, iterations=int(iterations)
        )
    elif operation == "Closing":
        # Closing = Dilation followed by Erosion (fills small holes and cracks)
        return cv2.morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, element, iterations=int(iterations)
        )
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


def preprocess_image(
    image_bgr: np.ndarray,
    filter_method: str = "Gaussian",
    filter_kernel: int = 5,
    filter_sigma: float = 1.0,
    thresh_method: str = "Otsu",
    thresh_val: int = 127,
    invert_polarity: bool = False,
    adaptive_method: str = "Gaussian",
    adaptive_block_size: int = 11,
    adaptive_c: int = 2,
    morph_operation: str = "Opening",
    morph_kernel: int = 3,
    morph_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Execute full classical preprocessing pipeline from BGR image to cleaned binary mask.

    Returns:
        (image_gray, clean_mask, pipeline_info)
    """
    # 1. Grayscale conversion
    gray = to_grayscale(image_bgr)

    # 2. Smoothing / Filtering
    filtered = apply_filter(
        gray, method=filter_method, kernel_size=filter_kernel, sigma=filter_sigma
    )

    # 3. Thresholding
    raw_mask, thresh_info = apply_threshold(
        filtered,
        method=thresh_method,
        thresh_val=thresh_val,
        invert=invert_polarity,
        adaptive_method=adaptive_method,
        block_size=adaptive_block_size,
        c_val=adaptive_c,
    )

    # 4. Morphological cleaning
    clean_mask = apply_morphology(
        raw_mask,
        operation=morph_operation,
        kernel_size=morph_kernel,
        iterations=morph_iterations,
    )

    pipeline_info: Dict[str, Any] = {
        "filter": {
            "method": filter_method,
            "kernel_size": filter_kernel,
            "sigma": filter_sigma,
        },
        "threshold": thresh_info,
        "morphology": {
            "operation": morph_operation,
            "kernel_size": morph_kernel,
            "iterations": morph_iterations,
        },
        "dimensions": {
            "height": int(image_bgr.shape[0]),
            "width": int(image_bgr.shape[1]),
            "channels": int(image_bgr.shape[2]) if image_bgr.ndim == 3 else 1,
        },
    }

    return gray, clean_mask, pipeline_info

"""
Visualization Module for Object Counting and Measurement Studio.

Provides image annotation rendering, contour overlays, bounding boxes,
object ID tags, and Matplotlib analytical charts (area comparison & intensity histograms).
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Streamlit and server environments
import matplotlib.pyplot as plt


# Palette of visually distinct high-contrast colors (BGR format for OpenCV)
PALETTE_BGR = [
    (46, 204, 113),   # Emerald Green
    (52, 152, 219),   # Sky Blue
    (155, 89, 182),   # Amethyst Purple
    (241, 196, 15),   # Sun Yellow
    (230, 126, 34),   # Carrot Orange
    (231, 76, 60),    # Alizarin Red
    (26, 188, 156),   # Turquoise
    (253, 121, 168),  # Rose Pink
    (9, 132, 227),    # Electron Blue
    (0, 184, 148),    # Mint Green
]


def draw_annotated_image(
    image_bgr: np.ndarray,
    contours: List[np.ndarray],
    measurements: List[Dict[str, Any]],
    show_contours: bool = True,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_centroids: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Render clean, professional visual annotations on top of the original image.

    Args:
        image_bgr: Base image in BGR format.
        contours: List of contour arrays.
        measurements: List of measurement dictionaries matching contours.
        show_contours: Draw colored contour boundaries.
        show_boxes: Draw bounding rectangles.
        show_labels: Draw object ID badge and area tag.
        show_centroids: Draw centroid point.
        line_thickness: Thickness of bounding boxes and contours.

    Returns:
        Annotated BGR image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Convert grayscale input to 3-channel BGR if needed
    if image_bgr.ndim == 2:
        annotated = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image_bgr.copy()

    for idx, (cnt, m) in enumerate(zip(contours, measurements)):
        color = PALETTE_BGR[idx % len(PALETTE_BGR)]
        obj_id = m.get("object_id", idx + 1)
        x = m.get("bbox_x", 0)
        y = m.get("bbox_y", 0)
        w = m.get("bbox_width_px", 0)
        h = m.get("bbox_height_px", 0)
        cx = int(m.get("centroid_x", x + w // 2))
        cy = int(m.get("centroid_y", y + h // 2))

        # 1. Draw contour boundary
        if show_contours:
            cv2.drawContours(annotated, [cnt], -1, color, line_thickness)

        # 2. Draw bounding rectangle
        if show_boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, line_thickness)

        # 3. Draw centroid dot
        if show_centroids:
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)  # Red center dot
            cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)  # White ring

        # 4. Draw object ID badge with high-contrast text pill
        if show_labels:
            label_text = f"#{obj_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thick = 1

            # Get text size for background banner
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
            badge_y1 = max(0, y - th - 6)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = x + tw + 8

            # Draw background pill
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                color,
                cv2.FILLED,
            )
            # Draw black text on colored pill
            cv2.putText(
                annotated,
                label_text,
                (badge_x1 + 4, badge_y2 - 3),
                font,
                font_scale,
                (20, 20, 20),
                font_thick,
                cv2.LINE_AA,
            )

    return annotated


def plot_area_distribution(
    measurements: List[Dict[str, Any]],
    unit_name: str = "px",
) -> plt.Figure:
    """
    Generate a clean Matplotlib bar chart comparing areas of detected objects.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=100)

    if not measurements:
        ax.text(
            0.5,
            0.5,
            "No objects detected to display area distribution",
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    is_calibrated = measurements[0].get("is_calibrated", False)
    area_key = "area_physical" if is_calibrated else "area_px"
    unit_label = f"({unit_name}²)" if is_calibrated else "(px²)"

    ids = [f"#{m['object_id']}" for m in measurements]
    areas = [m[area_key] for m in measurements]

    # Bar colors
    colors = [
        f"#{PALETTE_BGR[i % len(PALETTE_BGR)][2]:02x}{PALETTE_BGR[i % len(PALETTE_BGR)][1]:02x}{PALETTE_BGR[i % len(PALETTE_BGR)][0]:02x}"
        for i in range(len(ids))
    ]

    bars = ax.bar(ids, areas, color=colors, edgecolor="#2d3436", linewidth=0.8, width=0.6)

    # Labels and aesthetics
    ax.set_title("Object Area Comparison", fontsize=12, fontweight="bold", pad=12, color="#1e272e")
    ax.set_xlabel("Object ID", fontsize=10, labelpad=8, color="#2f3640")
    ax.set_ylabel(f"Area {unit_label}", fontsize=10, labelpad=8, color="#2f3640")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate value atop each bar if total objects <= 25
    if len(bars) <= 25:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}" if height < 1000 else f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
                color="#2d3436",
            )

    plt.xticks(rotation=45 if len(ids) > 12 else 0, fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    return fig


def plot_grayscale_histogram(
    image_gray: np.ndarray,
    threshold_value: Optional[float] = None,
) -> plt.Figure:
    """
    Generate an intensity distribution histogram (0-255) for the grayscale image.
    Optionally shows the selected threshold line.
    """
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)

    if image_gray is None or image_gray.size == 0:
        ax.text(0.5, 0.5, "No image data available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    hist, bins = np.histogram(image_gray.flatten(), bins=256, range=[0, 256])

    ax.fill_between(bins[:-1], hist, color="#74b9ff", alpha=0.6, step="mid")
    ax.plot(bins[:-1], hist, color="#0984e3", linewidth=1.2, drawstyle="steps-mid")

    if threshold_value is not None and isinstance(threshold_value, (int, float)):
        ax.axvline(
            x=threshold_value,
            color="#d63031",
            linestyle="--",
            linewidth=1.8,
            label=f"Threshold: {threshold_value:.1f}",
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title("Grayscale Pixel Intensity Histogram", fontsize=12, fontweight="bold", pad=10, color="#1e272e")
    ax.set_xlabel("Pixel Intensity (0 = Black, 255 = White)", fontsize=10, labelpad=6, color="#2f3640")
    ax.set_ylabel("Pixel Frequency", fontsize=10, labelpad=6, color="#2f3640")
    ax.set_xlim([0, 255])
    ax.grid(axis="both", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    plt.xticks(np.arange(0, 257, 32), fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    return fig

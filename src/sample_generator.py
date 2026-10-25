"""
Sample Image Generator for Object Counting and Measurement Studio.

Generates realistic synthetic test images with known counts and shapes:
1. coins.png - Multiple distinct circular coins of different radii
2. geometric_shapes.png - Geometric primitives (circle, square, rectangle, triangle, star)
3. pills_capsules.png - Pharmaceutical pill/capsule shapes
4. hardware_washers.png - Washers/nuts with central holes (to verify RETR_EXTERNAL ignores inner holes)
5. touching_objects.png - Clusters of touching items for watershed testing
"""

import os
from typing import Dict
import cv2
import numpy as np


def generate_coins_image() -> np.ndarray:
    """
    Generate an image of 7 circular coins on a textured tabletop-like background.
    """
    h, w = 480, 640
    # Clean light-gray background with subtle gradient
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    for r in range(h):
        img[r, :] = np.clip(240 - int(r * 0.05), 210, 255)

    # Coins definitions: (cx, cy, radius, color_bgr)
    coins = [
        (100, 120, 42, (180, 160, 50)),   # Golden coin
        (230, 100, 36, (195, 195, 195)), # Silver coin
        (380, 130, 48, (180, 160, 50)),  # Large golden
        (520, 110, 32, (160, 120, 80)),  # Bronze coin
        (160, 310, 50, (180, 160, 50)),  # Large golden
        (320, 320, 38, (195, 195, 195)), # Medium silver
        (480, 290, 44, (160, 120, 80)),  # Medium bronze
    ]

    for cx, cy, radius, color in coins:
        # Draw outer coin circle
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
        # Inner rim highlight
        cv2.circle(img, (cx, cy), radius - 4, (int(color[0]*0.85), int(color[1]*0.85), int(color[2]*0.85)), 2, cv2.LINE_AA)
        # Center emboss
        cv2.circle(img, (cx, cy), 12, (int(color[0]*1.1), int(color[1]*1.1), int(color[2]*1.1)), -1, cv2.LINE_AA)

    return img


def generate_geometric_shapes_image() -> np.ndarray:
    """
    Generate image with 6 distinct geometric shapes: Circle, Square, Rectangle, Triangle, Ellipse, Hexagon.
    Ideal for testing circularity, aspect ratio, and perimeter calculations.
    """
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 248

    # 1. Circle
    cv2.circle(img, (110, 120), 45, (46, 204, 113), -1, cv2.LINE_AA)

    # 2. Square (Aspect ratio ~1.0)
    cv2.rectangle(img, (240, 75), (330, 165), (52, 152, 219), -1)

    # 3. Rectangle (Aspect ratio != 1.0)
    cv2.rectangle(img, (430, 90), (570, 150), (155, 89, 182), -1)

    # 4. Triangle (Polygon)
    tri_pts = np.array([[110, 260], [60, 370], [160, 370]], np.int32)
    cv2.fillPoly(img, [tri_pts], (230, 126, 34))

    # 5. Ellipse
    cv2.ellipse(img, (285, 315), (60, 35), 25, 0, 360, (231, 76, 60), -1, cv2.LINE_AA)

    # 6. Regular Hexagon
    hex_cx, hex_cy, hex_r = 500, 315, 50
    hex_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_pts = np.array([
        [int(hex_cx + hex_r * np.cos(a)), int(hex_cy + hex_r * np.sin(a))]
        for a in hex_angles
    ], np.int32)
    cv2.fillPoly(img, [hex_pts], (241, 196, 15))

    return img


def generate_pills_image() -> np.ndarray:
    """
    Generate image with 8 medicinal tablets and capsule shapes.
    """
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 235

    pills = [
        # Circle pills
        ("circle", (100, 100), 28, (68, 189, 50)),
        ("circle", (260, 110), 32, (225, 112, 85)),
        ("circle", (420, 95), 26, (9, 132, 227)),
        ("circle", (540, 130), 30, (108, 92, 231)),
        # Capsule/oval pills
        ("oval", (120, 290), (55, 24, 0), (232, 67, 147)),
        ("oval", (290, 270), (50, 22, -30), (0, 184, 148)),
        ("oval", (440, 310), (48, 20, 45), (253, 203, 110)),
        ("oval", (550, 320), (52, 22, 15), (9, 132, 227)),
    ]

    for p in pills:
        ptype = p[0]
        if ptype == "circle":
            _, center, rad, color = p
            cv2.circle(img, center, rad, color, -1, cv2.LINE_AA)
            cv2.circle(img, center, rad - 3, (255, 255, 255), 1, cv2.LINE_AA)
        elif ptype == "oval":
            _, center, (axis_major, axis_minor, angle), color = p
            cv2.ellipse(img, center, (axis_major, axis_minor), angle, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, center, (axis_major - 3, axis_minor - 3), angle, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def generate_hardware_washers_image() -> np.ndarray:
    """
    Generate image with 5 hardware washers/nuts possessing prominent internal holes.
    Crucial for IPCV syllabus to prove cv2.RETR_EXTERNAL only detects external contours,
    so holes are not counted as separate items!
    """
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 242

    washers = [
        (130, 130, 52, 22, (99, 110, 114)),
        (320, 110, 44, 18, (120, 120, 120)),
        (510, 140, 60, 26, (80, 80, 85)),
        (200, 320, 56, 24, (90, 95, 100)),
        (440, 310, 48, 20, (110, 110, 115)),
    ]

    for cx, cy, outer_r, inner_r, color in washers:
        # Outer metallic body
        cv2.circle(img, (cx, cy), outer_r, color, -1, cv2.LINE_AA)
        # Inner hole (matches background color 242)
        cv2.circle(img, (cx, cy), inner_r, (242, 242, 242), -1, cv2.LINE_AA)
        # Edge highlights
        cv2.circle(img, (cx, cy), outer_r, (45, 52, 54), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), inner_r, (45, 52, 54), 2, cv2.LINE_AA)

    return img


def generate_touching_objects_image() -> np.ndarray:
    """
    Generate image with 2 separate objects and 2 touching pairs.
    Demonstrates standard contour detection (counts touching as 1) vs Watershed splitting!
    """
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 245

    color = (70, 70, 70)

    # Isolated object 1
    cv2.circle(img, (110, 120), 40, color, -1, cv2.LINE_AA)

    # Isolated object 2
    cv2.circle(img, (110, 330), 45, color, -1, cv2.LINE_AA)

    # Touching pair 1 (Two disks kissing with 4px overlap)
    cv2.circle(img, (320, 200), 45, color, -1, cv2.LINE_AA)
    cv2.circle(img, (395, 200), 45, color, -1, cv2.LINE_AA)

    # Touching triple
    cv2.circle(img, (490, 330), 38, color, -1, cv2.LINE_AA)
    cv2.circle(img, (555, 330), 38, color, -1, cv2.LINE_AA)

    return img


def save_all_sample_images(output_dir: str) -> Dict[str, str]:
    """
    Generate and save all synthetic sample images to the assets directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = {
        "Coins & Disks (7 Objects)": ("coins.png", generate_coins_image()),
        "Geometric Shapes (6 Objects)": ("geometric_shapes.png", generate_geometric_shapes_image()),
        "Pharmaceutical Pills (8 Objects)": ("pills_capsules.png", generate_pills_image()),
        "Hardware Washers with Holes (5 Objects)": ("hardware_washers.png", generate_hardware_washers_image()),
        "Touching Objects for Watershed": ("touching_objects.png", generate_touching_objects_image()),
    }

    saved_paths = {}
    for label, (filename, img) in samples.items():
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        saved_paths[label] = filepath

    return saved_paths


if __name__ == "__main__":
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sample_images")
    paths = save_all_sample_images(assets_dir)
    print(f"Generated {len(paths)} sample images in: {assets_dir}")

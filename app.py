"""
Object Counting and Measurement Studio Using Image Processing and Computer Vision
Author: IPCV Laboratory Student Mini-Project
Framework: Streamlit & OpenCV
"""

import os
import io
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Core processing modules
from src.preprocessing import preprocess_image, to_grayscale
from src.detection import detect_contours, segment_watershed
from src.measurement import measure_all_objects, compute_summary_statistics
from src.visualization import (
    draw_annotated_image,
    plot_area_distribution,
    plot_grayscale_histogram,
)
from src.export import (
    encode_image_png,
    measurements_to_csv,
    generate_json_report,
    measurements_to_dataframe,
)
from src.sample_generator import save_all_sample_images


# ---------------------------------------------------------
# Page Configuration & UI Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Object Counting & Measurement Studio | IPCV Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for university laboratory presentation aesthetic
st.markdown(
    """
    <style>
        /* Base typography & theme */
        .main {
            background-color: #fafbfc;
            color: #1e293b;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Metric card container */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: left;
            transition: all 0.2s ease-in-out;
        }
        .metric-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }
        .metric-title {
            color: #64748b;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-sub {
            color: #94a3b8;
            font-size: 0.78rem;
            margin-top: 4px;
        }
        
        /* Workflow badges */
        .workflow-step {
            display: inline-block;
            background-color: #f1f5f9;
            color: #334155;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 8px;
            border: 1px solid #e2e8f0;
        }
        .workflow-step-active {
            background-color: #0284c7;
            color: #ffffff;
            border-color: #0284c7;
        }
        
        /* Experiment badge */
        .lab-badge {
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 4px;
            border: 1px solid #bae6fd;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Asset Loading Helper
# ---------------------------------------------------------
@st.cache_resource
def ensure_sample_images() -> Dict[str, str]:
    """Ensure synthetic sample images are created and return paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "assets", "sample_images")
    if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
        return save_all_sample_images(sample_dir)
    return {
        "Coins & Disks (7 Objects)": os.path.join(sample_dir, "coins.png"),
        "Geometric Shapes (6 Objects)": os.path.join(sample_dir, "geometric_shapes.png"),
        "Pharmaceutical Pills (8 Objects)": os.path.join(sample_dir, "pills_capsules.png"),
        "Hardware Washers with Holes (5 Objects)": os.path.join(sample_dir, "hardware_washers.png"),
        "Touching Objects for Watershed": os.path.join(sample_dir, "touching_objects.png"),
    }


sample_dict = ensure_sample_images()


# ---------------------------------------------------------
# Application Header & Syllabus Context
# ---------------------------------------------------------
st.title("Object Counting and Measurement Studio")
st.markdown(
    "**Image Processing and Computer Vision (IPCV) Laboratory Mini-Project** | "
    "Classical Computer Vision Pipeline for Segmentation, Counting, and Morphometry"
)

# Workflow indicator
st.markdown(
    """
    <div>
        <span class="workflow-step workflow-step-active">1. Upload Image</span>
        <span class="workflow-step workflow-step-active">2. Preprocess & Threshold</span>
        <span class="workflow-step workflow-step-active">3. Detect & Separate</span>
        <span class="workflow-step workflow-step-active">4. Geometric Measurement</span>
        <span class="workflow-step workflow-step-active">5. Export Report</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------
# Sidebar Controls: Image Selection & Pipeline Parameters
# ---------------------------------------------------------
st.sidebar.header("1. Input Image Source")

input_mode = st.sidebar.radio(
    "Choose Image Mode:",
    ["Use Built-in Sample Image", "Upload My Own Image"],
    index=0,
    help="Select a benchmark academic test image or upload your own JPG/PNG/BMP file.",
)

image_bgr: Optional[np.ndarray] = None
image_filename: str = "sample_image.png"
image_filesize_kb: float = 0.0

if input_mode == "Use Built-in Sample Image":
    selected_sample_label = st.sidebar.selectbox(
        "Select Academic Benchmark Image:",
        list(sample_dict.keys()),
        index=0,
    )
    sample_path = sample_dict[selected_sample_label]
    if os.path.exists(sample_path):
        image_bgr = cv2.imread(sample_path)
        image_filename = os.path.basename(sample_path)
        image_filesize_kb = round(os.path.getsize(sample_path) / 1024.0, 1)
    else:
        st.sidebar.error("Sample image file not found on disk.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image File (JPG, PNG, BMP):",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image containing separated objects for counting and measurement.",
    )
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                st.sidebar.error("Error: Could not decode uploaded image file. It may be corrupted.")
            else:
                image_filename = uploaded_file.name
                image_filesize_kb = round(len(file_bytes) / 1024.0, 1)
        except Exception as e:
            st.sidebar.error(f"Error loading image: {str(e)}")

# Sidebar Reset Action
if st.sidebar.button("🔄 Reset Parameters to Default", use_container_width=True):
    st.rerun()

st.sidebar.divider()

# ---------------------------------------------------------
# Sidebar: Step 2 Preprocessing Controls
# ---------------------------------------------------------
st.sidebar.subheader("2. Preprocessing & Thresholding")

with st.sidebar.expander("Filtering & Smoothing", expanded=False):
    filter_method = st.selectbox(
        "Smoothing Filter:",
        ["Gaussian", "Median", "Box", "None"],
        index=0,
        help="Removes high-frequency camera sensor noise before thresholding.",
    )
    filter_kernel = st.slider(
        "Filter Kernel Size (px):",
        min_value=3,
        max_value=15,
        value=5,
        step=2,
        help="Must be an odd integer defining the spatial neighborhood.",
    )
    filter_sigma = 1.0
    if filter_method == "Gaussian":
        filter_sigma = st.slider(
            "Gaussian Sigma (σ):",
            min_value=0.2,
            max_value=4.0,
            value=1.0,
            step=0.1,
            help="Standard deviation of the Gaussian spatial kernel.",
        )

with st.sidebar.expander("Thresholding & Polarity", expanded=True):
    thresh_method = st.selectbox(
        "Segmentation Method:",
        ["Otsu", "Manual", "Adaptive"],
        index=0,
        help="Otsu automatically finds optimal intra-class threshold. Manual allows fine-tuning.",
    )

    thresh_val = 127
    if thresh_method == "Manual":
        thresh_val = st.slider(
            "Manual Intensity Cut-off (0-255):",
            min_value=0,
            max_value=255,
            value=127,
            step=1,
        )

    adaptive_method = "Gaussian"
    adaptive_block_size = 11
    adaptive_c = 2
    if thresh_method == "Adaptive":
        adaptive_method = st.selectbox("Adaptive Type:", ["Gaussian", "Mean"])
        adaptive_block_size = st.slider(
            "Adaptive Window Size:", min_value=3, max_value=51, value=15, step=2
        )
        adaptive_c = st.slider("Constant C:", min_value=-15, max_value=15, value=2, step=1)

    # Automatic polarity check recommendation
    default_invert = False
    if image_bgr is not None:
        temp_gray = to_grayscale(image_bgr)
        # If mean intensity is high (> 128), background is bright, objects are dark -> invert
        if np.mean(temp_gray) > 128:
            default_invert = True

    invert_polarity = st.checkbox(
        "Invert Polarity (Foreground is Darker)",
        value=default_invert,
        help="Check this if the objects are dark and the background is light.",
    )

with st.sidebar.expander("Morphological Operations", expanded=False):
    morph_op = st.selectbox(
        "Morphological Operation:",
        ["Opening", "Closing", "Erosion", "Dilation", "None"],
        index=0,
        help="Opening removes speckle noise; Closing seals internal object pinholes.",
    )
    morph_kernel = st.slider("Morphology Kernel Size:", min_value=1, max_value=11, value=3, step=2)
    morph_iter = st.slider("Morphology Iterations:", min_value=1, max_value=4, value=1, step=1)

st.sidebar.divider()

# ---------------------------------------------------------
# Sidebar: Step 3 Detection & Watershed Controls
# ---------------------------------------------------------
st.sidebar.subheader("3. Contour Detection & Filtering")

min_area_val = st.sidebar.slider(
    "Minimum Object Area (px²):",
    min_value=10,
    max_value=5000,
    value=80,
    step=10,
    help="Filters out tiny noise fragments and thresholding residue.",
)

max_area_val = st.sidebar.number_input(
    "Maximum Object Area (px²):",
    min_value=100,
    max_value=5_000_000,
    value=1_000_000,
    step=1000,
    help="Filters out whole-image borders or background regions.",
)

use_watershed = st.sidebar.checkbox(
    "Enable Watershed Segmentation",
    value=False,
    help="Separates touching or clustered objects using distance-transform peaks.",
)

watershed_ratio = 0.5
if use_watershed:
    watershed_ratio = st.sidebar.slider(
        "Watershed Peak Sensitivity (0.1 - 0.9):",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher values require sharper center peaks to declare separate objects.",
    )

st.sidebar.divider()

# ---------------------------------------------------------
# Sidebar: Step 4 Physical Calibration (Optional)
# ---------------------------------------------------------
st.sidebar.subheader("4. Physical Calibration (Optional)")

enable_calibration = st.sidebar.checkbox(
    "Enable Real-World Calibration",
    value=False,
    help="Converts pixel measurements to real physical units (mm, cm, in) using a known reference.",
)

pixels_per_unit = 1.0
unit_name = "px"

if enable_calibration:
    col_cal1, col_cal2 = st.sidebar.columns(2)
    with col_cal1:
        known_length = st.number_input(
            "Known Length:", min_value=0.1, max_value=1000.0, value=20.0, step=1.0
        )
    with col_cal2:
        unit_name = st.selectbox("Unit:", ["mm", "cm", "in"], index=0)

    known_pixels = st.sidebar.number_input(
        "Equivalent Length in Pixels (px):",
        min_value=1.0,
        max_value=5000.0,
        value=100.0,
        step=5.0,
        help="Number of image pixels corresponding to the known reference length.",
    )
    if known_length > 0:
        pixels_per_unit = known_pixels / known_length
        st.sidebar.caption(
            f"Scale Factor: **{pixels_per_unit:.2f} px/{unit_name}** (1 {unit_name}² = {pixels_per_unit**2:.1f} px²)"
        )
    else:
        pixels_per_unit = 1.0

st.sidebar.divider()

# ---------------------------------------------------------
# Sidebar: Step 5 Display & Annotation Overlays
# ---------------------------------------------------------
st.sidebar.subheader("5. Visual Overlays")
show_contours = st.sidebar.checkbox("Draw Contour Outlines", value=True)
show_boxes = st.sidebar.checkbox("Draw Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Draw Object ID Badges", value=True)
show_centroids = st.sidebar.checkbox("Draw Centroids", value=True)


# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
if image_bgr is None:
    st.info("👈 Please select a sample image or upload an image file from the sidebar to begin.")
    st.stop()

# 1. Run Preprocessing Pipeline
start_time = time.time()
image_gray, binary_mask, prep_info = preprocess_image(
    image_bgr=image_bgr,
    filter_method=filter_method,
    filter_kernel=filter_kernel,
    filter_sigma=filter_sigma,
    thresh_method=thresh_method,
    thresh_val=thresh_val,
    invert_polarity=invert_polarity,
    adaptive_method=adaptive_method,
    adaptive_block_size=adaptive_block_size,
    adaptive_c=adaptive_c,
    morph_operation=morph_op,
    morph_kernel=morph_kernel,
    morph_iterations=morph_iter,
)

# 2. Run Object Detection
if use_watershed:
    detected_contours, ws_markers, det_stats = segment_watershed(
        image_bgr=image_bgr,
        binary_mask=binary_mask,
        min_distance_ratio=watershed_ratio,
        min_area=min_area_val,
        max_area=max_area_val,
    )
else:
    detected_contours, det_stats = detect_contours(
        binary_mask=binary_mask,
        min_area=min_area_val,
        max_area=max_area_val,
    )

# 3. Calculate Morphological Measurements
measurements = measure_all_objects(
    contours=detected_contours,
    pixels_per_unit=pixels_per_unit if enable_calibration else 1.0,
    unit_name=unit_name if enable_calibration else "px",
)

summary_stats = compute_summary_statistics(
    measurements,
    unit_name=unit_name if enable_calibration else "px",
)

# 4. Render Visual Annotations
annotated_bgr = draw_annotated_image(
    image_bgr=image_bgr,
    contours=detected_contours,
    measurements=measurements,
    show_contours=show_contours,
    show_boxes=show_boxes,
    show_labels=show_labels,
    show_centroids=show_centroids,
)

elapsed_ms = round((time.time() - start_time) * 1000.0, 1)

# Format RGB images for Streamlit rendering
img_rgb_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
img_rgb_annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# Dashboard KPI Summary Cards
# ---------------------------------------------------------
kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Objects</div>
            <div class="metric-value">{summary_stats['total_count']}</div>
            <div class="metric-sub">Detected contours</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    area_unit_str = f"{unit_name}²" if enable_calibration else "px²"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Area</div>
            <div class="metric-value">{summary_stats['avg_area']:.1f}</div>
            <div class="metric-sub">Min: {summary_stats['min_area']:.1f} | Max: {summary_stats['max_area']:.1f} {area_unit_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    perim_unit_str = unit_name if enable_calibration else "px"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Perimeter</div>
            <div class="metric-value">{summary_stats['avg_perimeter']:.1f}</div>
            <div class="metric-sub">Boundary length ({perim_unit_str})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Image Dimensions</div>
            <div class="metric-value">{image_bgr.shape[1]}×{image_bgr.shape[0]}</div>
            <div class="metric-sub">{image_bgr.shape[2]} Ch | {image_filesize_kb} KB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[4]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Compute Latency</div>
            <div class="metric-value">{elapsed_ms} ms</div>
            <div class="metric-sub">OpenCV CPU Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if summary_stats["total_count"] == 0:
    st.warning(
        "⚠️ **No objects detected.** Try adjusting the Minimum Object Area slider, selecting 'Invert Polarity', or switching threshold methods in the sidebar."
    )


# ---------------------------------------------------------
# Tabbed Workspace: Inspection, Data Table, Charts, Syllabus, Export
# ---------------------------------------------------------
tab_visual, tab_table, tab_charts, tab_syllabus, tab_export = st.tabs(
    [
        "🖼️ Visual Inspection",
        "📋 Measurement Data Table",
        "📊 Statistical Analysis",
        "🎓 IPCV Syllabus Alignment",
        "💾 File Exports & Reports",
    ]
)

# ---------------------------------------------------------
# Tab 1: Visual Inspection Panels
# ---------------------------------------------------------
with tab_visual:
    st.subheader("Step-by-Step Image Transformation Pipeline")
    v_col1, v_col2, v_col3 = st.columns(3)

    with v_col1:
        st.markdown("**1. Original Input Image**")
        st.image(img_rgb_original, use_container_width=True)
        st.caption(f"Source: `{image_filename}` | {image_bgr.shape[1]}×{image_bgr.shape[0]} px")

    with v_col2:
        st.markdown(f"**2. Preprocessed Binary Mask ({thresh_method})**")
        st.image(binary_mask, use_container_width=True, clamp=True)
        st.caption(
            f"Filter: {filter_method} ({filter_kernel}x{filter_kernel}) | "
            f"Morphology: {morph_op} | Polarity: {'Inverted' if invert_polarity else 'Standard'}"
        )

    with v_col3:
        st.markdown(f"**3. Annotated Result ({summary_stats['total_count']} Objects)**")
        st.image(img_rgb_annotated, use_container_width=True)
        st.caption("Colored contours, bounding boxes, centroid dots, and object IDs")


# ---------------------------------------------------------
# Tab 2: Measurement Data Table
# ---------------------------------------------------------
with tab_table:
    st.subheader("Geometric Morphometry Measurements")
    if measurements:
        df_measurements = measurements_to_dataframe(measurements)
        st.dataframe(
            df_measurements,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "📌 **Formulas Used:** "
            "• Area: `cv2.contourArea` | "
            "• Perimeter: `cv2.arcLength(cnt, True)` | "
            "• Aspect Ratio: `Width / Height` | "
            "• Equiv. Diameter: `2 * sqrt(Area / π)` | "
            "• Circularity: `4π * Area / Perimeter²` (Circle = 1.0) | "
            "• Centroid: Moments `(M10/M00, M01/M00)`"
        )
        if enable_calibration:
            st.info(
                f"📏 **Physical Calibration Active:** Scaled by {pixels_per_unit:.2f} px/{unit_name}. "
                "Physical areas and lengths are derived from calibrated image geometry."
            )
        else:
            st.caption(
                "ℹ️ All measurements currently represent raw image-space pixel dimensions. Enable 'Real-World Calibration' in the sidebar to scale into physical units."
            )
    else:
        st.info("No measurement data available because no objects passed the current detection filters.")


# ---------------------------------------------------------
# Tab 3: Charts & Statistical Analysis
# ---------------------------------------------------------
with tab_charts:
    st.subheader("Statistical Distributions and Intensity Histograms")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Object Area Comparison")
        fig_area = plot_area_distribution(
            measurements, unit_name=unit_name if enable_calibration else "px"
        )
        st.pyplot(fig_area)

    with chart_col2:
        st.markdown("##### Grayscale Intensity Histogram")
        eff_thresh = prep_info["threshold"].get("effective_threshold")
        thresh_scalar = eff_thresh if isinstance(eff_thresh, (int, float)) else None
        fig_hist = plot_grayscale_histogram(image_gray, threshold_value=thresh_scalar)
        st.pyplot(fig_hist)


# ---------------------------------------------------------
# Tab 4: IPCV Laboratory Syllabus & Concepts
# ---------------------------------------------------------
with tab_syllabus:
    st.subheader("Academic IPCV Laboratory Mapping")
    st.markdown(
        """
        This software directly demonstrates foundational curriculum experiments prescribed in standard 
        university **Image Processing and Computer Vision (IPCV)** Laboratory syllabi:
        """
    )

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown(
            """
            #### 1. Point Processing & Spatial Filtering
            - **Grayscale Conversion**: Weighted channel luminance reduction ($Y = 0.299R + 0.587G + 0.114B$).
            - **Gaussian Smoothing**: 2D convolution with Gaussian kernel to attenuate additive white Gaussian noise.
            - **Median Filtering**: Non-linear ranking filter, exceptional at suppressing impulse (salt-and-pepper) noise without blurring sharp object boundaries.

            #### 2. Image Thresholding & Segmentation
            - **Global Thresholding**: Partitioning grayscale intensity histogram into foreground and background classes.
            - **Otsu's Automatic Thresholding**: Minimizes intra-class variance ($\sigma_w^2$) and maximizes between-class variance ($\sigma_b^2$) across pixel distributions.
            - **Adaptive Local Thresholding**: Dynamically computes threshold using Gaussian-weighted local neighborhoods.
            """
        )

    with exp_col2:
        st.markdown(
            """
            #### 3. Mathematical Morphology
            - **Erosion & Dilation**: Dual fundamental operations using rectangular or elliptical structuring elements.
            - **Morphological Opening**: Erosion followed by dilation; eliminates small background noise bridges.
            - **Morphological Closing**: Dilation followed by erosion; bridges narrow cracks and seals pinholes.

            #### 4. Boundary Representation & Morphometry
            - **Topological Contour Extraction**: Border following algorithm (`cv2.RETR_EXTERNAL`).
            - **Geometric Invariants**: Area via Green's theorem (`cv2.contourArea`), perimeter (`cv2.arcLength`), isoperimetric quotient (circularity), and spatial moments ($M_{10}/M_{00}, M_{01}/M_{00}$).
            - **Distance Transform & Watershed**: Topological marker-driven segmentation for decomposing overlapping convex bodies.
            """
        )


# ---------------------------------------------------------
# Tab 5: Exports & Downloadable Deliverables
# ---------------------------------------------------------
with tab_export:
    st.subheader("Generate & Download Project Artifacts")
    st.markdown(
        "Download processed visual outputs, numerical data tables, and complete academic experiment reports."
    )

    down_col1, down_col2, down_col3, down_col4 = st.columns(4)

    # 1. Annotated Image Download
    with down_col1:
        st.markdown("##### Annotated Image")
        annotated_png_bytes = encode_image_png(annotated_bgr)
        st.download_button(
            label="⬇️ Download Result (PNG)",
            data=annotated_png_bytes,
            file_name=f"annotated_{os.path.splitext(image_filename)[0]}.png",
            mime="image/png",
            use_container_width=True,
        )

    # 2. Binary Mask Download
    with down_col2:
        st.markdown("##### Binary Mask")
        mask_png_bytes = encode_image_png(binary_mask)
        st.download_button(
            label="⬇️ Download Mask (PNG)",
            data=mask_png_bytes,
            file_name=f"mask_{os.path.splitext(image_filename)[0]}.png",
            mime="image/png",
            use_container_width=True,
        )

    # 3. CSV Measurement Table Download
    with down_col3:
        st.markdown("##### Measurement Table")
        csv_data = measurements_to_csv(measurements)
        st.download_button(
            label="⬇️ Download Data (CSV)",
            data=csv_data,
            file_name=f"measurements_{os.path.splitext(image_filename)[0]}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 4. JSON Laboratory Report Download
    with down_col4:
        st.markdown("##### JSON Full Report")
        calib_dict = {
            "is_calibrated": enable_calibration,
            "unit": unit_name if enable_calibration else "px",
            "pixels_per_unit": float(pixels_per_unit) if enable_calibration else 1.0,
        }
        det_params = {
            "min_area": min_area_val,
            "max_area": max_area_val,
            "use_watershed": use_watershed,
            "watershed_ratio": watershed_ratio if use_watershed else None,
        }
        json_report_str = generate_json_report(
            measurements=measurements,
            summary_stats=summary_stats,
            image_metadata={
                "filename": image_filename,
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
                "channels": int(image_bgr.shape[2]),
                "filesize_kb": image_filesize_kb,
            },
            preprocessing_params=prep_info,
            detection_params=det_params,
            calibration_info=calib_dict,
        )
        st.download_button(
            label="⬇️ Download Report (JSON)",
            data=json_report_str,
            file_name=f"report_{os.path.splitext(image_filename)[0]}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.write("")
    with st.expander("📄 Preview Generated JSON Laboratory Report", expanded=False):
        st.code(json_report_str, language="json")


# ---------------------------------------------------------
# Application Footer
# ---------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
        <strong>Object Counting and Measurement Studio</strong> | Built with Python 3.11, OpenCV & Streamlit<br>
        Developed for University IPCV Laboratory Demonstration & Viva Examination
    </div>
    """,
    unsafe_allow_html=True,
)

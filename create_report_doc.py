"""
Script to generate the college mini-project report .docx file
formatted exactly according to the user's reference format, in Times New Roman.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def create_report():
    doc = docx.Document()

    # 1. Page Margins (Standard 1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # 2. Configure Default Style to Times New Roman 12pt
    style_normal = doc.styles["Normal"]
    font = style_normal.font
    font.name = "Times New Roman"
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # Ensure XML elements enforce Times New Roman font for Asian / Complex script too
    rFonts = style_normal.element.rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman")

    # Helper function to add paragraph with Times New Roman formatting
    def add_para(text="", bold=False, italic=False, size=12, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=6):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if text:
            run = p.add_run(text)
            run.bold = bold
            run.italic = italic
            run.font.name = "Times New Roman"
            run.font.size = Pt(size)
            run.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_tech_item(tech_name, description):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        run_name = p.add_run(tech_name + ": ")
        run_name.bold = True
        run_name.font.name = "Times New Roman"
        run_name.font.size = Pt(12)

        run_desc = p.add_run(description)
        run_desc.font.name = "Times New Roman"
        run_desc.font.size = Pt(12)
        return p

    # =========================================================
    # PAGE 1: TITLE / COVER PAGE (Matching user's reference)
    # =========================================================
    add_para(space_after=40)
    add_para(space_after=40)

    # Subject Title
    add_para("Image Processing and", bold=False, size=26, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_para("Computer Vision", bold=False, size=26, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)

    add_para(space_after=30)

    # Mini Project Label
    add_para("Mini Project", bold=False, size=24, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

    add_para(space_after=40)
    add_para(space_after=30)

    # Student Names & Roll Numbers (Matching reference layout)
    add_para("Meet Dawda – 6001724004", size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    add_para("Vedant Dave – 60017240057", size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    add_para("Nish Furia – 60017240098", size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    add_para("Shlok Jain – 60017240110", size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

    # End of Page 1 with explicit Times New Roman run
    p_pb = doc.add_paragraph()
    p_pb.paragraph_format.space_before = Pt(0)
    p_pb.paragraph_format.space_after = Pt(0)
    r_pb = p_pb.add_run()
    r_pb.font.name = "Times New Roman"
    r_pb.add_break(docx.enum.text.WD_BREAK.PAGE)

    # =========================================================
    # PAGE 2: REPORT CONTENT (Matching user's reference layout)
    # =========================================================
    # Project Title Header
    p_title = add_para("OBJECT COUNTING AND MEASUREMENT STUDIO USING IMAGE PROCESSING AND COMPUTER VISION", bold=True, size=12, space_after=12)

    # 1. INTRODUCTION
    add_heading_1("1. INTRODUCTION")
    add_para(
        "Object Counting and Measurement Studio is an interactive web-based computer vision application designed to automate the detection, enumeration, and geometric quantification of physical objects in digital images. Developed using Python, OpenCV, and Streamlit, the application applies classical digital image processing techniques to segment individual items from the background, calculate spatial and calibrated real-world measurements, and visualize results through dynamic charts. The system supports benchmark test imagery, custom image uploads, interactive parameter tuning for spatial filtering and morphological operations, and provides instant data exports in PNG, CSV, and JSON formats."
    )

    # 2. OBJECTIVES
    add_heading_1("2. OBJECTIVES")
    add_para("The main objectives of the Object Counting and Measurement Studio project are:")

    objectives = [
        "To develop a web-based computer vision application for detecting and counting discrete objects.",
        "To implement classical preprocessing methods including grayscale conversion, Gaussian smoothing, and median filtering.",
        "To perform image segmentation using manual thresholding, Otsu's intra-class variance minimization, and adaptive thresholding.",
        "To apply morphological operations (erosion, dilation, opening, and closing) to eliminate noise and bridge gaps.",
        "To extract external contours using the Suzuki-Abe border following algorithm without falsely counting internal holes.",
        "To calculate geometric measurements including contour area, perimeter, bounding-box dimensions, aspect ratio, equivalent circular diameter, and isoperimetric circularity.",
        "To demonstrate the practical application of Image Processing and Computer Vision (IPCV) laboratory concepts with comprehensive visual dashboards and data exports.",
    ]
    for idx, obj in enumerate(objectives, start=1):
        p_obj = doc.add_paragraph()
        p_obj.paragraph_format.left_indent = Inches(0.4)
        p_obj.paragraph_format.space_before = Pt(1)
        p_obj.paragraph_format.space_after = Pt(3)
        p_obj.paragraph_format.line_spacing = 1.15
        r_num = p_obj.add_run(f"{idx}. ")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(12)
        r_text = p_obj.add_run(obj)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(12)

    # 3. TECHNOLOGY STACK
    add_heading_1("3. TECHNOLOGY STACK")
    add_tech_item("Python 3.11", "Serves as the core programming language for implementing algorithms, modular logic, and data handling.")
    add_tech_item("OpenCV (opencv-python)", "Provides foundational computer vision functions for color conversions, spatial convolution, morphological transformations, contour discovery, and watershed segmentation.")
    add_tech_item("Streamlit", "Constructs the responsive web user interface, parameter sliders, visual image comparison tabs, and file download controllers.")
    add_tech_item("NumPy", "Performs high-performance multi-dimensional array operations, matrix manipulation, and pixel intensity filtering.")
    add_tech_item("Pandas", "Structures calculated object morphometry records into clean tabular dataframes and serializes data to CSV format.")
    add_tech_item("Matplotlib", "Generates statistical distribution charts, including object area comparison bar charts and 256-bin pixel intensity histograms.")
    add_tech_item("Pillow (PIL)", "Supports image loading, color validation, and format conversions.")
    add_tech_item("Pytest", "Executes automated test suites across preprocessing, detection, and measurement modules to guarantee algorithm reliability.")

    # 4. SYSTEM DESIGN AND IMPLEMENTATION
    add_heading_1("4. SYSTEM DESIGN AND IMPLEMENTATION")
    add_para(
        "The Object Counting and Measurement Studio consists of modular components that collaborate sequentially to transform raw image matrices into calibrated analytical measurements."
    )

    add_heading_2("4.1 Image Preprocessing Module")
    add_para(
        "The preprocessing module prepares the input image by converting three-channel BGR inputs to single-channel 8-bit grayscale images using perceptual luminance weighting. To eliminate high-frequency sensor noise without degrading object boundaries, the user can apply Gaussian smoothing or median filtering with configurable odd kernel dimensions."
    )

    add_heading_2("4.2 Segmentation and Thresholding Module")
    add_para(
        "The segmentation module isolates foreground objects from the background. It provides Otsu's global automatic thresholding, manual thresholding, and adaptive Gaussian/Mean thresholding for non-uniform illumination. A polarity inversion control ensures reliable binarization whether objects are lighter or darker than the background. Morphological opening and closing are subsequently applied to remove small background speckles and seal pinholes."
    )

    add_heading_2("4.3 Object Detection and Watershed Splitting Module")
    add_para(
        "Contour detection is implemented using cv2.RETR_EXTERNAL to identify external boundaries. This ensures that items with internal holes (such as hardware washers) are counted exactly once. Contours below a minimum area or above a maximum area are filtered out. For touching or kissing objects, an optional distance-transform and marker-controlled watershed segmentation splits clustered items into individual components."
    )

    add_heading_2("4.4 Geometric Measurement and Morphometry Module")
    add_para(
        "For each detected contour, the measurement module calculates the following metrics:"
    )

    metrics = [
        ("Contour Area", "Calculated via Green's theorem in discrete vector fields using cv2.contourArea."),
        ("Perimeter", "Computed as closed polygonal arc length using cv2.arcLength."),
        ("Bounding Box & Aspect Ratio", "Calculates width, height, and ratio (Width/Height) to evaluate elongation."),
        ("Equivalent Circular Diameter", "Computed as 2 * sqrt(Area / pi) to determine the diameter of an equivalent circular disk."),
        ("Isoperimetric Circularity", "Calculated as 4 * pi * Area / Perimeter^2 (1.0 for circles, ~0.785 for squares)."),
        ("Spatial Centroid", "Derived from spatial image moments M10/M00 and M01/M00."),
        ("Physical Calibration", "Converts raw pixel metrics into millimeters, centimeters, or inches using a known reference factor."),
    ]
    for m_title, m_desc in metrics:
        p_m = doc.add_paragraph()
        p_m.paragraph_format.left_indent = Inches(0.4)
        p_m.paragraph_format.space_before = Pt(1)
        p_m.paragraph_format.space_after = Pt(2)
        p_m.paragraph_format.line_spacing = 1.15
        r_bullet = p_m.add_run("● ")
        r_bullet.font.name = "Times New Roman"
        r_bullet.font.size = Pt(10)
        r_btitle = p_m.add_run(f"{m_title}: ")
        r_btitle.bold = True
        r_btitle.font.name = "Times New Roman"
        r_btitle.font.size = Pt(12)
        r_bdesc = p_m.add_run(m_desc)
        r_bdesc.font.name = "Times New Roman"
        r_bdesc.font.size = Pt(12)

    add_heading_2("4.5 Visualization and Export Module")
    add_para(
        "The studio generates visual overlays highlighting contours, bounding boxes, centroid dots, and numbered ID tags. Users can download annotated PNG images, binary segmentation masks, Pandas CSV measurement spreadsheets, and complete JSON laboratory reports."
    )

    # 5. WORKING OF THE APPLICATION
    add_heading_1("5. WORKING OF THE APPLICATION")
    add_para("The application follows these sequential steps:")

    steps = [
        "Image Input: The user uploads an image file (JPG, PNG, BMP) or selects one of the 5 built-in academic benchmark images.",
        "Grayscale Conversion: The three-channel color image is transformed into single-channel grayscale.",
        "Spatial Filtering: Gaussian or median filtering is applied to suppress noise while preserving edges.",
        "Thresholding & Binarization: The image is segmented using Otsu, manual, or adaptive thresholding with optional polarity inversion.",
        "Morphological Cleaning: Opening or closing operations eliminate residual speckle noise and bridge gaps.",
        "Contour Detection & Splitting: External contours are extracted using cv2.RETR_EXTERNAL, or separated using watershed segmentation if objects touch.",
        "Measurement Computation: Geometric dimensions (area, perimeter, circularity, equivalent diameter, centroids) are computed for each object.",
        "Output Generation: The application updates summary KPI cards, displays side-by-side visual panels, plots area distribution bar charts and intensity histograms, and provides download options.",
    ]
    for idx, step_text in enumerate(steps, start=1):
        p_step = doc.add_paragraph()
        p_step.paragraph_format.left_indent = Inches(0.4)
        p_step.paragraph_format.space_before = Pt(1)
        p_step.paragraph_format.space_after = Pt(3)
        p_step.paragraph_format.line_spacing = 1.15
        r_num = p_step.add_run(f"{idx}. ")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(12)
        r_t = p_step.add_run(step_text)
        r_t.font.name = "Times New Roman"
        r_t.font.size = Pt(12)

    add_para(
        "This workflow unifies image acquisition, spatial filtering, mathematical morphology, topological contour extraction, and calibrated measurement into a single, cohesive academic platform."
    )

    # 6. OUTPUTS
    add_heading_1("6. OUTPUTS")
    add_para(
        "[Insert Screenshots of the Application Here: Studio Dashboard, Visual Inspection Tab showing Original Image, Binary Mask, and Annotated Result, Measurement Data Table, and Statistical Analysis Charts]"
    )
    add_para(space_after=20)

    # Challenges / Limitations paragraph (matching format of page 6 in sample)
    add_para(
        "The application may face challenges when handling non-planar objects with significant perspective distortion, non-uniform illumination with directional shadows, and severe physical occlusions where items overlap by more than fifty percent. Performance depends on proper contrast between foreground objects and the background plane. Future improvements could include incorporating subpixel edge localization for industrial metrology, automated perspective rectification using chessboard calibration patterns, oriented bounding box fitting (cv2.minAreaRect), and multi-spectral color clustering in HSV and CIE-Lab color spaces."
    )

    # 8. CONCLUSION
    add_heading_1("8. CONCLUSION")
    add_para(
        "Object Counting and Measurement Studio is an educational and practical computer vision application that demonstrates key Image Processing and Computer Vision (IPCV) laboratory concepts within an interactive web platform. Developed using Python, OpenCV, Streamlit, and NumPy, it combines classical image filtering, optimal thresholding, mathematical morphology, topological contour extraction, and geometric morphometry. The application provides reliable, reproducible, and deterministic counting and measurement on standard CPU hardware without requiring cloud dependencies or black-box neural networks. Overall, the project provides a comprehensive understanding of digital image processing principles, practical algorithm implementation, and software engineering for computer vision applications."
    )

    # PROJECT LINKS
    add_heading_1("PROJECT LINKS")
    p_gh = doc.add_paragraph()
    p_gh.paragraph_format.space_before = Pt(2)
    p_gh.paragraph_format.space_after = Pt(4)
    r_gh_label = p_gh.add_run("GitHub Repository: ")
    r_gh_label.bold = True
    r_gh_label.font.name = "Times New Roman"
    r_gh_label.font.size = Pt(12)
    r_gh_link = p_gh.add_run("https://github.com/your-username/object-counting-measurement-studio")
    r_gh_link.font.name = "Times New Roman"
    r_gh_link.font.size = Pt(12)
    r_gh_link.underline = True

    p_live = doc.add_paragraph()
    p_live.paragraph_format.space_before = Pt(2)
    p_live.paragraph_format.space_after = Pt(4)
    r_live_label = p_live.add_run("Live Application: ")
    r_live_label.bold = True
    r_live_label.font.name = "Times New Roman"
    r_live_label.font.size = Pt(12)
    r_live_link = p_live.add_run("https://object-counting-measurement-studio.streamlit.app")
    r_live_link.font.name = "Times New Roman"
    r_live_link.font.size = Pt(12)
    r_live_link.underline = True

    output_path = "Object_Counting_and_Measurement_Studio_Report.docx"
    doc.save(output_path)
    print(f"Report generated successfully: {output_path}")


if __name__ == "__main__":
    create_report()

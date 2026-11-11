
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import tempfile
import os

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolov8n_crowd_best.pt"

IMAGE_DIR = ROOT / "outputs" / "images" / "streamlit"
VIDEO_DIR = ROOT / "outputs" / "videos" / "streamlit"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

LOW_MAX = 5
MEDIUM_MAX = 15

st.set_page_config(
    page_title="CrowdVision | Smart Crowd Detection",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- GREEN WHITE DESIGN ----------------

st.markdown("""
<style>

/* =========================================
   GLOBAL DESIGN
========================================= */

.stApp {
    background: #f3faf5;
    color: #243b2d;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 18px;
}

.block-container {
    max-width: 1400px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* =========================================
   TYPOGRAPHY - LARGER LETTERS
========================================= */

h1 {
    color: #064e3b !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    line-height: 1.25 !important;
}

h2 {
    color: #14532d !important;
    font-size: 2.15rem !important;
    font-weight: 750 !important;
}

h3 {
    color: #166534 !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

p, .stMarkdown p {
    font-size: 18px !important;
    line-height: 1.75 !important;
    color: #334b3a;
}

li {
    font-size: 17px !important;
    line-height: 1.8 !important;
}

label {
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* =========================================
   PREMIUM SIDEBAR
========================================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #064e3b 0%,
        #14532d 55%,
        #15803d 100%
    );
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] h2 {
    font-size: 27px !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] p {
    font-size: 15px !important;
    line-height: 1.6 !important;
}

[data-testid="stSidebar"] label {
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 7px 4px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 9px;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.22);
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25);
}

/* =========================================
   HERO HEADER
========================================= */

.hero {
    background: linear-gradient(
        120deg,
        #064e3b,
        #15803d,
        #16a34a
    );
    padding: 42px 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(6,78,59,0.18);
}

.hero h1, .hero h2 {
    color: #ffffff !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    margin-bottom: 12px;
}

.hero p {
    color: #e6ffed !important;
    font-size: 19px !important;
    line-height: 1.7 !important;
}

/* =========================================
   DASHBOARD METRIC CARDS
========================================= */

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ccebd4;
    border-radius: 19px;
    padding: 25px 23px;
    min-height: 135px;
    box-shadow: 0 7px 20px rgba(22,101,52,0.07);
}

[data-testid="stMetricLabel"] {
    color: #477653 !important;
    font-size: 16px !important;
    font-weight: 650 !important;
}

[data-testid="stMetricValue"] {
    color: #14532d !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

/* =========================================
   FEATURE CARDS
========================================= */

.feature-card {
    background: #ffffff;
    border: 1px solid #d1ead7;
    border-left: 5px solid #16a34a;
    border-radius: 18px;
    padding: 28px;
    min-height: 240px;
    box-shadow: 0 8px 22px rgba(22,101,52,0.07);
    transition: 0.2s ease;
}

.feature-card h3 {
    color: #15803d !important;
    font-size: 1.6rem !important;
}

.feature-card p {
    font-size: 17px !important;
}

/* =========================================
   BUTTONS
========================================= */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(120deg, #15803d, #166534);
    color: white !important;
    border: none;
    border-radius: 12px;
    min-height: 49px;
    padding: 10px 23px;
    font-size: 17px !important;
    font-weight: 700 !important;
    box-shadow: 0 5px 12px rgba(22,101,52,0.12);
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #064e3b;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(22,101,52,0.2);
}

/* =========================================
   UPLOAD AREA
========================================= */

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #65a875;
    border-radius: 17px;
    padding: 20px;
}

[data-testid="stFileUploader"] p {
    font-size: 17px !important;
}

/* =========================================
   INPUTS AND SLIDERS
========================================= */

.stSlider [data-baseweb="slider"] {
    padding-top: 12px;
    padding-bottom: 12px;
}

[data-baseweb="select"] {
    font-size: 17px !important;
}

/* =========================================
   ALERTS
========================================= */

[data-testid="stAlert"] {
    border-radius: 14px;
    padding: 15px 18px;
}

[data-testid="stAlert"] p {
    font-size: 17px !important;
}

/* =========================================
   DATA TABLE
========================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #ccebd4;
    border-radius: 14px;
    overflow: hidden;
}

/* =========================================
   FOOTER
========================================= */

.footer {
    text-align: center;
    color: #52745c;
    padding: 25px;
    font-size: 15px;
    line-height: 1.8;
}

/* =========================================
   DIVIDERS
========================================= */

hr {
    border-color: #d5eadd;
    margin-top: 25px;
    margin-bottom: 25px;
}

</style>
\n""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

names = model.names
items = names.items() if isinstance(names, dict) else enumerate(names)

person_id = next(
    (int(i) for i, n in items if str(n).lower() in ["person", "people"]),
    0
)


# ---------------- CROWD CLASSIFICATION ----------------

def crowd_level(count):
    if count <= LOW_MAX:
        return "Low"
    if count <= MEDIUM_MAX:
        return "Medium"
    return "High"


def show_level(count):
    level = crowd_level(count)
    if level == "Low":
        st.success("🟢 LOW CROWD LEVEL")
    elif level == "Medium":
        st.warning("🟡 MEDIUM CROWD LEVEL")
    else:
        st.error("🔴 HIGH CROWD LEVEL")
    return level


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<div style="text-align:center;padding:20px 5px;">
    <div style="font-size:48px;">👥</div>
    <h2 style="color:white!important;margin:0;">CrowdVision</h2>
    <p style="color:#dcfce7!important;font-size:12px;">
        INTELLIGENT CROWD DETECTION
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Home", "Image Detection", "Video Detection", "About Project"]
)

st.sidebar.divider()

st.sidebar.markdown("""
**SYSTEM DETAILS**

- Model: YOLOv8n
- Device: CPU
- Class: Person
- Interface: Streamlit
""")

st.sidebar.markdown("""
<div style="padding-top:30px;color:#dcfce7!important;
font-size:12px;">
MCA Major Project<br>
Crowd Detection in Public Transport
</div>
""", unsafe_allow_html=True)


# ---------------- HOME ----------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to CrowdVision 👋</h1>
        <p>
        Intelligent crowd monitoring for public transport,
        powered by Artificial Intelligence and YOLOv8.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.title("Dashboard Overview")

    st.write(
        "Detect people, calculate crowd counts, and analyze "
        "crowd levels from images and videos."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Detection Model", "YOLOv8n")
    c2.metric("Execution Device", "CPU")
    c3.metric("Detection Class", "Person")

    st.divider()
    st.subheader("Explore Our Features")

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="feature-card">
        <h3>🖼️ Image Detection</h3>
        <p>Upload an image and detect people automatically.</p>
        <p>✓ Bounding box visualization</p>
        <p>✓ Automatic person counting</p>
        <p>✓ Crowd-level classification</p>
        <p>✓ Download annotated image</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="feature-card">
        <h3>🎥 Video Detection</h3>
        <p>Analyze video frames using the trained model.</p>
        <p>✓ Frame-wise detection</p>
        <p>✓ Crowd count statistics</p>
        <p>✓ Crowd count graph</p>
        <p>✓ Download processed video</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Crowd Classification")

    x, y, z = st.columns(3)
    x.success("🟢 LOW\n\n0–5 People")
    y.warning("🟡 MEDIUM\n\n6–15 People")
    z.error("🔴 HIGH\n\n16+ People")

    st.caption(
        "These are configurable demonstration thresholds, "
        "not official public safety limits."
    )


# ---------------- IMAGE DETECTION ----------------

elif page == "Image Detection":

    st.title("🖼️ Image Detection")
    st.write("Upload an image to detect and count people.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    confidence = st.slider(
        "Detection Confidence",
        0.10, 0.90, 0.25, 0.05
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        if st.button("🚀 Detect People", type="primary"):

            with st.spinner("Analyzing image..."):

                result = model.predict(
                    source=image,
                    imgsz=320,
                    conf=confidence,
                    device="cpu",
                    classes=[person_id],
                    verbose=False
                )[0]

                count = len(result.boxes) if result.boxes is not None else 0

                annotated = cv2.cvtColor(
                    result.plot(), cv2.COLOR_BGR2RGB
                )

                level = crowd_level(count)

                output = IMAGE_DIR / (
                    "crowd_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".jpg"
                )

                Image.fromarray(annotated).save(output)

            st.divider()
            st.subheader("Detection Results")

            c1, c2 = st.columns(2)
            c1.metric("People Detected", count)
            c2.metric("Crowd Level", level)

            show_level(count)

            st.subheader("Detected Image")
            st.image(annotated, use_container_width=True)

            if result.boxes is not None and len(result.boxes):
                confidence_values = result.boxes.conf.cpu().numpy()

                df = pd.DataFrame({
                    "Person": range(1, count + 1),
                    "Confidence": [
                        round(float(v), 4) for v in confidence_values
                    ]
                })

                st.subheader("Detection Confidence")
                st.dataframe(df, use_container_width=True)

            else:
                st.info("No people detected.")

            with open(output, "rb") as f:
                st.download_button(
                    "⬇️ Download Annotated Image",
                    data=f.read(),
                    file_name=output.name,
                    mime="image/jpeg"
                )


# ---------------- VIDEO DETECTION ----------------

elif page == "Video Detection":

    st.title("🎥 Video Detection")
    st.write("Upload a short video for frame-wise crowd analysis.")

    st.info(
        "Video processing runs on CPU. Begin with a short video "
        "to keep processing time manageable."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

    confidence = st.slider(
        "Detection Confidence",
        0.10, 0.90, 0.25, 0.05,
        key="video_confidence"
    )

    max_frames = st.slider(
        "Maximum Frames",
        30, 600, 150, 30
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🚀 Process Video", type="primary"):

            temp_path = None
            cap = None
            writer = None

            try:
                suffix = Path(uploaded_video.name).suffix or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp:
                    temp.write(uploaded_video.getbuffer())
                    temp_path = temp.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    raise ValueError("Unable to open uploaded video.")

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    fps = 25.0

                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    raise ValueError("Video contains no readable frames.")

                limit = min(total, max_frames)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output = VIDEO_DIR / (
                    "processed_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".mp4"
                )

                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise ValueError("Could not create processed video.")

                progress = st.progress(0)
                status = st.empty()
                counts = []

                for i in range(limit):

                    success, frame = cap.read()
                    if not success:
                        break

                    result = model.predict(
                        source=frame,
                        imgsz=320,
                        conf=confidence,
                        device="cpu",
                        classes=[person_id],
                        verbose=False
                    )[0]

                    count = len(result.boxes) if result.boxes is not None else 0
                    counts.append(count)

                    writer.write(result.plot())

                    progress.progress((i + 1) / limit)
                    status.write(f"Processing frame {i + 1} / {limit}")

                cap.release()
                writer.release()
                cap = None
                writer = None

                if not counts:
                    raise ValueError("No frames could be processed.")

                average = round(sum(counts) / len(counts), 2)
                maximum = max(counts)
                final_count = counts[-1]

                st.success("Video processing completed!")

                st.subheader("Crowd Statistics")

                a, b, c = st.columns(3)
                a.metric("Frames Processed", len(counts))
                b.metric("Average People", average)
                c.metric("Maximum People", maximum)

                st.subheader("Last Frame")
                st.metric("People Detected", final_count)
                show_level(final_count)

                df = pd.DataFrame({
                    "Frame": range(1, len(counts) + 1),
                    "Person Count": counts
                })

                st.subheader("Frame-wise Crowd Analysis")
                st.line_chart(df.set_index("Frame"))

                st.subheader("Processed Video")
                st.video(str(output))

                with open(output, "rb") as f:
                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=f.read(),
                        file_name=output.name,
                        mime="video/mp4"
                    )

                st.download_button(
                    "⬇️ Download Frame Count CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="crowd_frame_counts.csv",
                    mime="text/csv"
                )

                if len(counts) < total:
                    st.warning(
                        f"Processed {len(counts)} of {total} frames "
                        "due to the selected frame limit or video read limit."
                    )

            except Exception as e:
                st.error(f"Video processing error: {e}")

            finally:
                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("📘 About CrowdVision")

    st.markdown("""
    <div class="hero">
        <h2>Crowd Detection in Public Transport</h2>
        <p>AI-powered people detection and crowd monitoring.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Objective")

    st.write("""
    This project detects people in public transport images and videos,
    calculates the number of detected people, and classifies crowd levels.
    """)

    st.subheader("Technology Stack")

    technologies = pd.DataFrame({
        "Technology": [
            "Python", "YOLOv8n", "OpenCV", "Streamlit",
            "NumPy", "Pandas", "Ultralytics"
        ],
        "Purpose": [
            "Programming Language",
            "Object Detection",
            "Image and Video Processing",
            "Web Interface",
            "Numerical Computing",
            "Data Analysis",
            "YOLO Framework"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Crowd Classification")
    st.success("LOW: 0–5 people")
    st.warning("MEDIUM: 6–15 people")
    st.error("HIGH: 16 or more people")

    st.caption(
        "These thresholds are demonstration values, not official "
        "public safety standards."
    )


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div class="footer">
    <b>CrowdVision</b><br>
    Crowd Detection in Public Transport<br>
    Built with Python, YOLOv8 and Streamlit | MCA Major Project
</div>
""", unsafe_allow_html=True)


st.markdown('<style>\n\n\n/* CROWDVISION LARGE NAVIGATION BUTTONS */\n\n[data-testid="stSidebar"] [role="radiogroup"] {\n    width: 100% !important;\n    gap: 13px !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label {\n    width: 100% !important;\n    min-height: 65px !important;\n    padding: 16px 18px !important;\n    margin-bottom: 11px !important;\n    border-radius: 15px !important;\n    display: flex !important;\n    align-items: center !important;\n    box-sizing: border-box !important;\n    background: rgba(255,255,255,0.12) !important;\n    transition: all 0.2s ease !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label p {\n    font-size: 20px !important;\n    font-weight: 700 !important;\n    line-height: 1.4 !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label:hover {\n    background: rgba(255,255,255,0.25) !important;\n    transform: translateX(4px);\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {\n    background: rgba(255,255,255,0.28) !important;\n    border: 1px solid rgba(255,255,255,0.45) !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label div[role="radio"] {\n    transform: scale(1.3);\n    margin-right: 8px;\n}\n\n</style>', unsafe_allow_html=True)


st.markdown('\n<style>\n\n/* CROWDVISION_DETECTION_PAGE_PREMIUM */\n\n/* Main page background and alignment */\n.stApp {\n    background: #f5faf6 !important;\n}\n\n.block-container {\n    max-width: 1350px !important;\n    padding: 2.5rem 3rem 3rem !important;\n}\n\n/* Detection page headings */\n.main h1 {\n    color: #064e3b !important;\n    font-size: 2.8rem !important;\n    font-weight: 850 !important;\n    letter-spacing: -1px !important;\n    padding-bottom: 15px;\n    border-bottom: 3px solid #d1ead7;\n    margin-bottom: 18px !important;\n}\n\n.main h2 {\n    color: #14532d !important;\n    font-size: 2rem !important;\n    font-weight: 750 !important;\n}\n\n.main h3 {\n    color: #166534 !important;\n    font-size: 1.5rem !important;\n    font-weight: 750 !important;\n}\n\n/* Main text */\n.main p {\n    font-size: 18px !important;\n    line-height: 1.7 !important;\n    color: #385344 !important;\n}\n\n/* Upload panel */\n[data-testid="stFileUploader"] {\n    background: #ffffff !important;\n    border: 2px dashed #52a56b !important;\n    border-radius: 20px !important;\n    padding: 22px !important;\n    box-shadow: 0 7px 22px rgba(20,83,45,0.07) !important;\n    margin: 12px 0 24px !important;\n}\n\n[data-testid="stFileUploader"] label {\n    color: #14532d !important;\n    font-size: 19px !important;\n    font-weight: 750 !important;\n}\n\n[data-testid="stFileUploaderDropzone"] {\n    background: #f1faf3 !important;\n    border: 1px solid #d2ead8 !important;\n    border-radius: 14px !important;\n    min-height: 100px !important;\n}\n\n[data-testid="stFileUploaderDropzone"] button {\n    background: #15803d !important;\n    color: white !important;\n    border-radius: 10px !important;\n    font-size: 16px !important;\n    font-weight: 700 !important;\n    border: none !important;\n}\n\n/* Slider labels and green accent */\n[data-testid="stSlider"] {\n    background: #ffffff;\n    border: 1px solid #d6eadb;\n    border-radius: 15px;\n    padding: 17px 20px 12px;\n    margin: 15px 0 25px;\n}\n\n[data-testid="stSlider"] label {\n    font-size: 18px !important;\n    color: #14532d !important;\n    font-weight: 750 !important;\n}\n\n[data-testid="stSlider"] [role="slider"] {\n    background: #15803d !important;\n    border-color: #15803d !important;\n}\n\n[data-testid="stSlider"] [data-baseweb="slider"] > div > div {\n    background: #16a34a !important;\n}\n\n/* Main action buttons */\n.stButton > button,\n.stDownloadButton > button {\n    min-height: 56px !important;\n    width: 100%;\n    border-radius: 13px !important;\n    background: linear-gradient(110deg, #166534, #16a34a) !important;\n    color: white !important;\n    font-size: 18px !important;\n    font-weight: 750 !important;\n    border: none !important;\n    box-shadow: 0 6px 16px rgba(22,101,52,0.15);\n    transition: all 0.2s ease;\n}\n\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n    background: #064e3b !important;\n    transform: translateY(-2px);\n    box-shadow: 0 9px 20px rgba(22,101,52,0.22);\n}\n\n/* Metrics */\n[data-testid="stMetric"] {\n    background: #ffffff !important;\n    border: 1px solid #cce8d2 !important;\n    border-top: 4px solid #16a34a !important;\n    border-radius: 17px !important;\n    padding: 23px !important;\n    box-shadow: 0 6px 20px rgba(20,83,45,0.07);\n}\n\n[data-testid="stMetricLabel"] {\n    color: #477653 !important;\n    font-size: 17px !important;\n    font-weight: 700 !important;\n}\n\n[data-testid="stMetricValue"] {\n    color: #14532d !important;\n    font-size: 2.4rem !important;\n    font-weight: 850 !important;\n}\n\n/* Information and result messages */\n[data-testid="stAlert"] {\n    border-radius: 15px !important;\n    padding: 17px 20px !important;\n    border-left: 5px solid #16a34a !important;\n}\n\n[data-testid="stAlert"] p {\n    font-size: 17px !important;\n}\n\n/* Uploaded and detected images */\n[data-testid="stImage"] img {\n    border-radius: 17px !important;\n    border: 1px solid #d4e9d9 !important;\n    box-shadow: 0 7px 22px rgba(20,83,45,0.09);\n}\n\n/* Video player */\n[data-testid="stVideo"] {\n    border-radius: 17px !important;\n    overflow: hidden !important;\n    border: 1px solid #d4e9d9 !important;\n    box-shadow: 0 7px 22px rgba(20,83,45,0.09);\n}\n\n/* Data tables */\n[data-testid="stDataFrame"] {\n    background: #ffffff;\n    border: 1px solid #d1ead7;\n    border-radius: 15px;\n    overflow: hidden;\n}\n\n/* Footer */\n.footer {\n    border-top: 1px solid #d1ead7;\n    margin-top: 35px;\n    padding: 28px 15px !important;\n    color: #477653 !important;\n    font-size: 16px !important;\n    line-height: 2 !important;\n    text-align: center;\n}\n\n/* Dividers */\nhr {\n    border-color: #d1ead7 !important;\n    margin: 28px 0 !important;\n}\n\n</style>\n', unsafe_allow_html=True)


st.markdown('\n<style>\n\n/* ==================================\n   CROWDVISION BURGUNDY THEME\n================================== */\n\n.stApp {\n    background: #fffafb !important;\n    color: #482432 !important;\n}\n\n.block-container {\n    background: #fffafb !important;\n}\n\n/* Main headings */\n.main h1,\n.main h2,\n.main h3,\nh1, h2, h3 {\n    color: #641b32 !important;\n}\n\n.main h1 {\n    font-weight: 850 !important;\n}\n\n.main p,\n.main li {\n    color: #583443 !important;\n}\n\n/* Sidebar */\n[data-testid="stSidebar"] {\n    background: linear-gradient(\n        180deg,\n        #451020 0%,\n        #641b32 55%,\n        #852b49 100%\n    ) !important;\n}\n\n[data-testid="stSidebar"] * {\n    color: #ffffff !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label {\n    background: rgba(255,255,255,0.10) !important;\n    border: 1px solid rgba(255,255,255,0.12) !important;\n    border-radius: 15px !important;\n    min-height: 65px !important;\n    padding: 16px 18px !important;\n    margin-bottom: 11px !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label p {\n    color: #ffffff !important;\n    font-size: 20px !important;\n    font-weight: 700 !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label:hover {\n    background: rgba(255,255,255,0.23) !important;\n}\n\n[data-testid="stSidebar"] hr {\n    border-color: rgba(255,255,255,0.3) !important;\n}\n\n/* Hero panels */\n.hero {\n    background: linear-gradient(\n        120deg,\n        #451020,\n        #641b32,\n        #8b2948\n    ) !important;\n    border-radius: 22px !important;\n    box-shadow: 0 10px 28px rgba(100,27,50,0.18) !important;\n}\n\n.hero h1,\n.hero h2 {\n    color: #ffffff !important;\n}\n\n.hero p {\n    color: #fce9ef !important;\n}\n\n/* Metric cards */\n[data-testid="stMetric"] {\n    background: #ffffff !important;\n    border: 1px solid #ead0d9 !important;\n    border-top: 4px solid #8b2948 !important;\n    border-radius: 18px !important;\n    box-shadow: 0 7px 20px rgba(100,27,50,0.07) !important;\n}\n\n[data-testid="stMetricLabel"] {\n    color: #82465a !important;\n}\n\n[data-testid="stMetricValue"] {\n    color: #641b32 !important;\n}\n\n/* Feature cards */\n.feature-card {\n    background: #ffffff !important;\n    border: 1px solid #ead0d9 !important;\n    border-left: 5px solid #8b2948 !important;\n    box-shadow: 0 7px 20px rgba(100,27,50,0.07) !important;\n}\n\n.feature-card h3 {\n    color: #8b2948 !important;\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button {\n    background: linear-gradient(\n        110deg,\n        #641b32,\n        #8b2948\n    ) !important;\n    color: #ffffff !important;\n    border: none !important;\n    border-radius: 13px !important;\n    font-weight: 750 !important;\n    box-shadow: 0 5px 15px rgba(100,27,50,0.16) !important;\n}\n\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n    background: #451020 !important;\n    color: #ffffff !important;\n}\n\n/* File uploader */\n[data-testid="stFileUploader"] {\n    background: #ffffff !important;\n    border: 2px dashed #b66a82 !important;\n    border-radius: 18px !important;\n    box-shadow: 0 6px 18px rgba(100,27,50,0.05) !important;\n}\n\n[data-testid="stFileUploader"] label {\n    color: #641b32 !important;\n}\n\n[data-testid="stFileUploaderDropzone"] {\n    background: #fcf2f5 !important;\n    border: 1px solid #ead0d9 !important;\n    border-radius: 14px !important;\n}\n\n[data-testid="stFileUploaderDropzone"] button {\n    background: #641b32 !important;\n    color: white !important;\n    border: none !important;\n}\n\n/* Sliders */\n[data-testid="stSlider"] {\n    background: #ffffff !important;\n    border: 1px solid #ead0d9 !important;\n    border-radius: 15px !important;\n}\n\n[data-testid="stSlider"] label {\n    color: #641b32 !important;\n}\n\n[data-testid="stSlider"] [role="slider"] {\n    background: #8b2948 !important;\n    border-color: #8b2948 !important;\n}\n\n[data-testid="stSlider"] [data-baseweb="slider"]\n> div > div {\n    background: #8b2948 !important;\n}\n\n/* Alerts */\n[data-testid="stAlert"] {\n    border-radius: 14px !important;\n}\n\n/* Image and video borders */\n[data-testid="stImage"] img,\n[data-testid="stVideo"] {\n    border: 1px solid #ead0d9 !important;\n    border-radius: 17px !important;\n    box-shadow: 0 6px 18px rgba(100,27,50,0.08) !important;\n}\n\n/* Tables */\n[data-testid="stDataFrame"] {\n    border: 1px solid #ead0d9 !important;\n    border-radius: 14px !important;\n}\n\n/* Footer and dividers */\n.footer {\n    color: #82465a !important;\n    border-color: #ead0d9 !important;\n}\n\nhr {\n    border-color: #ead0d9 !important;\n}\n\n</style>\n', unsafe_allow_html=True)


st.markdown('\n<style>\n\n/* CROWDVISION_LUXURY_BURGUNDY_V2 */\n\n/* ===== PAGE BACKGROUND ===== */\n\n.stApp {\n    background:\n        radial-gradient(ellipse at 10% 5%,\n            rgba(221, 165, 183, 0.20), transparent 32%),\n        radial-gradient(ellipse at 95% 35%,\n            rgba(193, 124, 147, 0.13), transparent 30%),\n        linear-gradient(135deg, #fff9fb 0%, #f8edf1 48%, #fff7f9 100%)\n        !important;\n    color: #492333 !important;\n}\n\n[data-testid="stAppViewContainer"] {\n    background: transparent !important;\n}\n\n[data-testid="stMain"] {\n    background: transparent !important;\n}\n\n.block-container {\n    max-width: 1450px !important;\n    padding: 2.5rem 3rem 3rem !important;\n    position: relative;\n}\n\n/* ===== FIX SIDEBAR WIDTH AND ALIGNMENT ===== */\n\n[data-testid="stSidebar"] {\n    width: 300px !important;\n    min-width: 300px !important;\n    flex: 0 0 300px !important;\n    background: linear-gradient(\n        160deg,\n        #3b0d20 0%,\n        #58152f 48%,\n        #792844 100%\n    ) !important;\n    border-right: 5px solid #a64b68 !important;\n    box-shadow: 7px 0 25px rgba(65, 12, 32, 0.13) !important;\n}\n\n[data-testid="stSidebar"] > div {\n    width: 100% !important;\n}\n\n[data-testid="stSidebar"] * {\n    color: #fff7fa !important;\n}\n\n[data-testid="stSidebar"] h1,\n[data-testid="stSidebar"] h2,\n[data-testid="stSidebar"] h3 {\n    color: #ffffff !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] {\n    width: 100% !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label {\n    width: 100% !important;\n    min-height: 62px !important;\n    padding: 15px 16px !important;\n    box-sizing: border-box !important;\n    border-radius: 14px !important;\n    background: rgba(255,255,255,0.09) !important;\n    margin: 8px 0 !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label p {\n    color: white !important;\n    font-size: 18px !important;\n    font-weight: 700 !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label:hover {\n    background: rgba(255,255,255,0.20) !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {\n    background: linear-gradient(\n        100deg, #a74465, #82304d\n    ) !important;\n    border: 1px solid rgba(255,255,255,0.35) !important;\n}\n\n[data-testid="stSidebar"] hr {\n    border-color: rgba(255,255,255,0.25) !important;\n}\n\n/* ===== PAGE HEADINGS ===== */\n\n.main h1 {\n    color: #58152f !important;\n    font-size: 2.8rem !important;\n    font-weight: 850 !important;\n    letter-spacing: -1px !important;\n    padding-bottom: 15px !important;\n    border-bottom: 3px solid #ead0da !important;\n}\n\n.main h2 {\n    color: #6d1e3b !important;\n    font-size: 2rem !important;\n    font-weight: 800 !important;\n}\n\n.main h3 {\n    color: #852b49 !important;\n    font-size: 1.5rem !important;\n    font-weight: 750 !important;\n}\n\n.main p,\n.main li {\n    color: #563748 !important;\n    font-size: 18px !important;\n    line-height: 1.75 !important;\n}\n\n/* ===== INFORMATION PANELS ===== */\n\n[data-testid="stAlert"] {\n    background: linear-gradient(\n        110deg, #f8e8ee, #fff6f8\n    ) !important;\n    border: 1px solid #e7c2d0 !important;\n    border-left: 6px solid #852b49 !important;\n    border-radius: 15px !important;\n    box-shadow: 0 6px 18px rgba(100,27,50,0.06) !important;\n}\n\n[data-testid="stAlert"] p {\n    color: #64243c !important;\n}\n\n/* ===== UPLOAD PANELS ===== */\n\n[data-testid="stFileUploader"] {\n    background: rgba(255,255,255,0.94) !important;\n    border: 2px dashed #a64b68 !important;\n    border-radius: 20px !important;\n    padding: 23px !important;\n    box-shadow: 0 10px 25px rgba(100,27,50,0.08) !important;\n    margin: 15px 0 25px !important;\n}\n\n[data-testid="stFileUploader"] label {\n    color: #641b32 !important;\n    font-size: 19px !important;\n    font-weight: 750 !important;\n}\n\n[data-testid="stFileUploaderDropzone"] {\n    background: linear-gradient(\n        120deg, #fff5f8, #f9eaf0\n    ) !important;\n    border: 1px solid #ebcbd6 !important;\n    border-radius: 15px !important;\n}\n\n[data-testid="stFileUploaderDropzone"] button {\n    background: #641b32 !important;\n    color: white !important;\n    border-radius: 10px !important;\n    font-weight: 700 !important;\n}\n\n/* ===== SLIDER CARDS ===== */\n\n[data-testid="stSlider"] {\n    background: rgba(255,255,255,0.96) !important;\n    border: 1px solid #e9cbd5 !important;\n    border-radius: 17px !important;\n    padding: 20px 22px 14px !important;\n    margin: 20px 0 26px !important;\n    box-shadow: 0 7px 20px rgba(100,27,50,0.06) !important;\n}\n\n[data-testid="stSlider"] label {\n    color: #641b32 !important;\n    font-size: 18px !important;\n    font-weight: 750 !important;\n}\n\n[data-testid="stSlider"] [role="slider"] {\n    background: #852b49 !important;\n    border-color: #852b49 !important;\n}\n\n[data-testid="stSlider"] [data-baseweb="slider"]\n> div > div {\n    background: #852b49 !important;\n}\n\n/* Remove unwanted dark backgrounds behind slider values */\n[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {\n    background: transparent !important;\n}\n\n/* ===== BUTTONS ===== */\n\n.stButton > button,\n.stDownloadButton > button {\n    background: linear-gradient(\n        110deg, #58152f, #8b2948\n    ) !important;\n    color: white !important;\n    border: none !important;\n    border-radius: 13px !important;\n    min-height: 54px !important;\n    font-size: 18px !important;\n    font-weight: 750 !important;\n    box-shadow: 0 7px 17px rgba(88,21,47,0.18) !important;\n    transition: all 0.2s ease !important;\n}\n\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n    background: linear-gradient(\n        110deg, #3b0d20, #641b32\n    ) !important;\n    transform: translateY(-2px) !important;\n}\n\n/* ===== METRIC CARDS ===== */\n\n[data-testid="stMetric"] {\n    background: #ffffff !important;\n    border: 1px solid #ead0da !important;\n    border-top: 5px solid #852b49 !important;\n    border-radius: 18px !important;\n    padding: 23px !important;\n    box-shadow: 0 8px 23px rgba(100,27,50,0.08) !important;\n}\n\n[data-testid="stMetricLabel"] {\n    color: #82465a !important;\n}\n\n[data-testid="stMetricValue"] {\n    color: #58152f !important;\n    font-weight: 850 !important;\n}\n\n/* ===== IMAGE AND VIDEO DISPLAY ===== */\n\n[data-testid="stImage"] img {\n    border-radius: 18px !important;\n    border: 2px solid #ead0da !important;\n    box-shadow: 0 10px 25px rgba(100,27,50,0.10) !important;\n}\n\n[data-testid="stVideo"] {\n    border: 2px solid #ead0da !important;\n    border-radius: 18px !important;\n    overflow: hidden !important;\n    box-shadow: 0 10px 25px rgba(100,27,50,0.10) !important;\n}\n\n/* ===== TABLES ===== */\n\n[data-testid="stDataFrame"] {\n    background: white !important;\n    border: 1px solid #ead0da !important;\n    border-radius: 15px !important;\n    overflow: hidden !important;\n}\n\n/* ===== FOOTER ===== */\n\n.footer {\n    background: linear-gradient(\n        100deg, #f8e9ef, #fff8fa\n    ) !important;\n    border-top: 1px solid #e8c8d4 !important;\n    border-radius: 15px !important;\n    padding: 25px !important;\n    margin-top: 35px !important;\n    color: #6d3349 !important;\n    text-align: center !important;\n}\n\n/* ===== DIVIDERS ===== */\n\nhr {\n    border-color: #e8c8d4 !important;\n}\n\n</style>\n', unsafe_allow_html=True)


st.markdown('\n<style>\n\n/* Make slider minimum and maximum labels readable */\n[data-testid="stSlider"] [data-testid="stTickBarMin"],\n[data-testid="stSlider"] [data-testid="stTickBarMax"] {\n    background: transparent !important;\n    color: #641b32 !important;\n    padding: 3px 0 !important;\n    border: none !important;\n    font-size: 15px !important;\n    font-weight: 650 !important;\n}\n\n/* Fix slider value displayed above the thumb */\n[data-testid="stSlider"] [role="slider"] {\n    background: #852b49 !important;\n    border: 2px solid #852b49 !important;\n}\n\n/* Prevent unwanted coloured backgrounds on slider labels */\n[data-testid="stSlider"] code,\n[data-testid="stSlider"] span {\n    background: transparent !important;\n    color: #641b32 !important;\n}\n\n/* Slider card */\n[data-testid="stSlider"] {\n    background: #ffffff !important;\n    border: 1px solid #e8c5d1 !important;\n    border-radius: 17px !important;\n    padding: 22px !important;\n}\n\n</style>\n', unsafe_allow_html=True)

# SLIDER_TRANSPARENT_LABELS_FIX

st.markdown('\n<style>\n\n/* Remove coloured background from slider min/max numbers */\n[data-testid="stSlider"] code,\n[data-testid="stSlider"] small,\n[data-testid="stSlider"] [data-testid="stTickBarMin"],\n[data-testid="stSlider"] [data-testid="stTickBarMax"],\n[data-testid="stSlider"] [data-testid="stTickBarMin"] *,\n[data-testid="stSlider"] [data-testid="stTickBarMax"] * {\n    background: transparent !important;\n    background-color: transparent !important;\n    color: #641b32 !important;\n    border: none !important;\n    box-shadow: none !important;\n    padding: 0 !important;\n    border-radius: 0 !important;\n}\n\n/* Keep slider numbers clearly visible */\n[data-testid="stSlider"] {\n    color: #641b32 !important;\n}\n\n</style>\n', unsafe_allow_html=True)

# HIDE_SLIDER_MIN_MAX_LABELS_V1

st.markdown('\n<style>\n\n/* Completely hide minimum and maximum slider numbers */\n[data-testid="stSlider"] [data-testid="stTickBarMin"],\n[data-testid="stSlider"] [data-testid="stTickBarMax"] {\n    display: none !important;\n    visibility: hidden !important;\n}\n\n/* Also hide any endpoint labels rendered as small text */\n[data-testid="stSlider"] [data-testid="stTickBarMin"] *,\n[data-testid="stSlider"] [data-testid="stTickBarMax"] * {\n    display: none !important;\n}\n\n</style>\n', unsafe_allow_html=True)

# FINAL HIDE SLIDER LIMIT LABELS

st.markdown('\n<style>\n\n/* Hide slider minimum and maximum labels */\n[data-testid="stSlider"] [data-testid*="TickBar"],\n[data-testid="stSlider"] [class*="TickBarMin"],\n[data-testid="stSlider"] [class*="TickBarMax"],\n[data-testid="stSlider"] .stSliderTickBarMin,\n[data-testid="stSlider"] .stSliderTickBarMax {\n    display: none !important;\n    visibility: hidden !important;\n    opacity: 0 !important;\n    height: 0 !important;\n}\n\n/* Keep selected slider value visible */\n[data-testid="stSlider"] [role="slider"] {\n    visibility: visible !important;\n}\n\n</style>\n', unsafe_allow_html=True)

"""
Streamlit Web Application: Real-Time Sign Language Recognition Using MediaPipe & Machine Learning
ML Lab Mini-Project Dashboard
"""

import os
import json
import logging
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import base64
import cv2
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="ASL Sign Recognition | ML Lab Project",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Dark Computer Vision Workspace Aesthetic (Matching Reference Design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600;1,700&family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Lora:ital,wght@0,400..700;1,400..700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography - Editorial Serif Primary */
    html, body, [class*="css"], .stMarkdown, .stText, p, label, li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        font-family: 'Newsreader', 'Lora', Georgia, 'Times New Roman', serif !important;
        -webkit-font-smoothing: antialiased;
        color: #d1d5db !important;
        font-size: 15px;
        line-height: 1.65;
    }

    /* Headings Hierarchy - Playfair Display Editorial Serif */
    h1, [data-testid="stHeading"] h1, .stHeading h1 {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: #f9fafb !important;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
    }

    h2, h3, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3,
    .stHeading h2, .stHeading h3, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: #f9fafb !important;
        margin-top: 18px !important;
        margin-bottom: 8px !important;
    }

    h4, h5, h6 {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #e5e7eb !important;
    }

    /* Deep Dark Charcoal Canvas */
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #111216 !important;
        color: #f3f4f6 !important;
    }

    /* Responsive, Compact Container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1150px !important;
    }

    /* Main Container */
    .main {
        background-color: transparent !important;
    }

    /* Header Bar */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Dark Workspace Sidebar */
    [data-testid="stSidebar"] {
        background-color: #14161b !important;
        border-right: 1px solid #242730 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #14161b !important;
    }

    /* Sidebar Project Name: Playfair Display 700, 20px */
    [data-testid="stSidebar"] h1 {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #f9fafb !important;
        margin-top: 8px !important;
        margin-bottom: 2px !important;
    }

    /* Sidebar Technical Section Labels: JetBrains Mono */
    [data-testid="stSidebar"] h3 {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #6b7280 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }

    /* Sidebar Navigation Container */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 3px !important;
    }

    /* Sidebar Radio: Clean Styling, Transparent Text Containers, No Yellow/White Boxes */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 15px !important;
        font-weight: 450 !important;
        color: #9ca3af !important;
        padding: 7px 12px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        background: transparent !important;
        background-color: transparent !important;
        margin-bottom: 2px !important;
    }

    /* ALL child elements inside radio labels (text, markdown, divs) must be transparent */
    [data-testid="stSidebar"] [data-testid="stRadio"] label *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:focus *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:active *,
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] * {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Elegant Subtle Hover on Sidebar Label */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #1c2027 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover span {
        color: #ffffff !important;
    }

    /* Active (Checked) Selected State: Subtle Clean Dark Active Background */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background-color: #1a1e26 !important;
        border: 1px solid #282d38 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) span {
        color: #f9fafb !important;
        font-weight: 600 !important;
    }

    /* Radio Circle Accent in Pure Red */
    [data-testid="stRadio"] input[type="radio"] {
        accent-color: #ef4444 !important;
    }

    /* Technical Workspace Header Panel */
    .header-card {
        background-color: #181a20;
        border: 1px solid #282b34;
        border-radius: 6px;
        padding: 18px 22px;
        margin-bottom: 20px;
        position: relative;
    }

    .header-card::before {
        content: "WORKSPACE • REAL-TIME ASL RECOGNITION";
        display: block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        color: #ef4444;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
        text-transform: uppercase;
    }

    .header-card h1 {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        color: #f9fafb !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin: 0 0 4px 0 !important;
    }

    .header-card p {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        color: #9ca3af !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        margin: 0 !important;
    }

    /* Metric Panels: Editorial Serif with Warm Red Highlights */
    .metric-card {
        background-color: #181a20;
        border: 1px solid #282b34;
        border-radius: 6px;
        padding: 14px 18px;
        box-shadow: none;
        text-align: left;
    }

    .metric-value {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #ef4444 !important;
        font-variant-numeric: tabular-nums !important;
        line-height: 1.15 !important;
    }

    .metric-label {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 12px !important;
        color: #9ca3af !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        margin-top: 4px !important;
        text-transform: uppercase !important;
    }

    /* Prediction Panel: Clean Terminal Detection Box */
    .prediction-badge {
        background-color: #181a20;
        border: 1px solid #282b34;
        border-radius: 6px;
        padding: 18px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 14px;
        box-shadow: none;
    }

    .predicted-letter {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 3.5rem;
        font-weight: 700;
        color: #ef4444;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .status-text {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 14px;
        font-weight: 500;
        color: #9ca3af;
        margin-top: 6px;
    }

    /* Technical Section Headers with Monospace Accents */
    .tech-label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #ef4444 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* Streamlit Native Metric Cards */
    [data-testid="stMetric"] {
        background-color: #181a20 !important;
        border: 1px solid #282b34 !important;
        border-radius: 6px !important;
        padding: 14px 18px !important;
        box-shadow: none !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        font-family: 'Playfair Display', 'Newsreader', Georgia, serif !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #ef4444 !important;
        line-height: 1.15 !important;
        font-variant-numeric: tabular-nums !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
    }

    /* Tables & Dataframes */
    [data-testid="stDataFrame"], .stDataFrame {
        border: 1px solid #282b34 !important;
        border-radius: 6px !important;
        background-color: #181a20 !important;
    }

    table {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 14px !important;
        border-collapse: collapse !important;
        color: #f3f4f6 !important;
    }

    th {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #9ca3af !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        background-color: #14161b !important;
        border-bottom: 1px solid #282b34 !important;
        padding: 8px 12px !important;
    }

    td {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #f3f4f6 !important;
        border-bottom: 1px solid #20222a !important;
        padding: 8px 12px !important;
    }

    /* Alerts and Notices */
    [data-testid="stAlert"] {
        background-color: #181a20 !important;
        border: 1px solid #282b34 !important;
        border-radius: 6px !important;
        color: #f3f4f6 !important;
    }

    [data-testid="stAlert"] p, [data-testid="stAlert"] span {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        font-size: 15px !important;
        color: #f3f4f6 !important;
    }

    /* Subtle Dark Dividers */
    hr {
        border-color: #242730 !important;
        margin: 20px 0 !important;
    }

    /* Clean Workspace Buttons */
    .stButton > button {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        border: 1px solid #2d313a !important;
        color: #f3f4f6 !important;
        background-color: #1c1e24 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        border-color: #ef4444 !important;
        color: #ef4444 !important;
        background-color: #22181a !important;
    }

    /* Selectboxes and Inputs */
    .stSelectbox label, .stRadio label {
        font-family: 'Newsreader', 'Lora', Georgia, serif !important;
        color: #e5e7eb !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #181a20 !important;
        border-color: #282b34 !important;
        color: #f3f4f6 !important;
        border-radius: 4px !important;
    }

    /* High Contrast Monospace Code Blocks */
    pre, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        background-color: #0c0d10 !important;
        border: 1px solid #242730 !important;
        border-radius: 4px !important;
        font-size: 13px !important;
    }

    pre *, [data-testid="stCodeBlock"] * {
        color: #f3f4f6 !important;
        font-family: 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }

    /* Inline Code Spans: JetBrains Mono with Subtle Highlight */
    code:not(pre code) {
        font-family: 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 12px !important;
        color: #f87171 !important;
        background-color: #1f1416 !important;
        border: 1px solid #3d2024 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATASET_INFO_FILE = os.path.join("data", "dataset_info.json")
LANDMARKS_FILE = os.path.join("data", "landmarks.csv")
DATASET_SUMMARY_FILE = os.path.join("results", "dataset_summary.json")
EXTRACTION_SUMMARY_FILE = os.path.join("results", "extraction_summary.json")
MODEL_COMPARISON_FILE = os.path.join("results", "model_comparison.json")
MODEL_METADATA_FILE = os.path.join("models", "model_metadata.json")
COMPARISON_CHART_FILE = os.path.join("results", "model_comparison_chart.png")
HAND_LANDMARKS_IMG = os.path.join("assets", "hand_landmarks.png")

WORD_DATASET_INFO_FILE = os.path.join("data", "wlasl", "word_dataset_info.json")
WORD_MODELS_DIR = os.path.join("models", "word_models")
WORD_METADATA_FILE = os.path.join(WORD_MODELS_DIR, "word_model_metadata.json")
WORD_COMPARISON_FILE = os.path.join("results", "word_results", "word_model_comparison.json")


@st.cache_resource
def get_predictor():
    """Initializes and caches the sign language predictor."""
    from src.prediction import SignLanguagePredictor
    return SignLanguagePredictor()


@st.cache_resource
def get_word_predictor():
    """Initializes and caches the word-level sign predictor."""
    from src.word_recognition import WordSignPredictor
    return WordSignPredictor()


def load_json(filepath: str) -> Optional[dict]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Sidebar Navigation
LOGO_FILE = os.path.join("assets", "logo.svg")
if os.path.exists(LOGO_FILE):
    st.sidebar.image(LOGO_FILE, width=105)
st.sidebar.title("ASL ML Project")
st.sidebar.caption("Real-Time Sign Language Recognition")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🏠 Home Dashboard", "📷 Live Recognition", "🔤 Words Recognition", "📊 Model Comparison", "📁 Dataset Explorer", "ℹ️ About Project"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

has_downloaded = os.path.exists(DATASET_INFO_FILE)
has_landmarks = os.path.exists(LANDMARKS_FILE)
has_models = os.path.exists(MODEL_METADATA_FILE)
has_word_models = os.path.exists(WORD_METADATA_FILE)

st.sidebar.write(f"Alphabet Dataset: {'✅ Ready' if has_downloaded else '⏳ Pending'}")
st.sidebar.write(f"Alphabet Landmarks: {'✅ Ready' if has_landmarks else '⏳ Pending'}")
st.sidebar.write(f"Alphabet Models: {'✅ Ready' if has_models else '⏳ Pending'}")
st.sidebar.write(f"Word Models (WLASL): {'✅ Ready' if has_word_models else '⏳ Pending'}")

st.sidebar.markdown("---")
st.sidebar.caption("ML Lab Mini-Project • BTech CSE/ECE")


# ==============================================================================
# 1. HOME DASHBOARD
# ==============================================================================
if nav_choice == "🏠 Home Dashboard":
    st.markdown("""
    <div class="header-card">
        <h1>Real-Time Sign Language Recognition</h1>
        <p>Using MediaPipe Hand Landmarking & Tuned Machine Learning Classifiers</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Load metadata if available
    comp_data = load_json(MODEL_COMPARISON_FILE)
    ext_data = load_json(EXTRACTION_SUMMARY_FILE)

    with col1:
        total_samples = ext_data.get("total_hands_detected", "N/A") if ext_data else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_samples}</div>
            <div class="metric-label">Landmark Samples</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        classes_count = len(ext_data.get("classes", [])) if ext_data else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{classes_count}</div>
            <div class="metric-label">Alphabet Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best_model = comp_data.get("best_model", "N/A") if comp_data else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.4rem;">{best_model}</div>
            <div class="metric-label">Top Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        best_acc = f"{comp_data.get('best_accuracy', 0.0)*100:.1f}%" if comp_data else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_acc}</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem Statement & Objectives
    st.subheader("🎯 Problem Statement & Objectives")
    st.markdown("""
    Deaf and hard-of-hearing individuals communicate primary concepts via sign language. Bridging the gap
    between non-signers and sign language users requires efficient, low-latency, and accessible assistive tools.
    
    **Core Objectives:**
    1. **Data Cleaning & Pipeline Isolation:** Eliminate data leakage using stratified train/test partitions and pipelines.
    2. **Landmark-Based Feature Extraction:** Extract 21 hand landmarks ($63$ spatial dimensions: $x, y, z$) via MediaPipe with scale and translation invariance.
    3. **Model Benchmark & Hyperparameter Tuning:** Train and compare strictly 4 classical algorithms:
       - **K-Nearest Neighbors (KNN)**
       - **Support Vector Machine (SVM)**
       - **Decision Tree**
       - **Random Forest**
    4. **Real-Time Interactive Interface:** Provide a live webcam demo and performance dashboard.
    """)

    st.markdown("---")
    st.subheader("🔄 End-to-End Project Workflow")
    st.markdown("""
    ```
    [Kaggle Dataset Download]
               │
               ▼
    [Image Validation & Cleaning]  ──► [Skip unreadable & duplicates]
               │
               ▼
    [MediaPipe Hand Landmarking]   ──► [21 Landmarks -> 63 Normalized Features (x, y, z)]
               │
               ▼
    [Stratified Split (80/20)]     ──► [Zero Data Leakage Guarantee]
               │
               ▼
    [Model Tuning & Cross-Val]     ──► [KNN, SVM, Decision Tree, Random Forest]
               │
               ▼
    [Held-out Test Evaluation]     ──► [Accuracy, Macro F1, Precision, Recall, CM]
               │
               ▼
    [Real-Time Webcam Recognition] ──► [Live OpenCV Frame -> Landmarks -> ML Classifier]
    ```
    """)

    st.markdown("---")
    st.subheader("🖐️ Hand Landmark Topology (21 Keypoints)")
    st.markdown("MediaPipe extracts **21 precise 3D spatial keypoints** (63 values: $x, y, z$) representing joint coordinates normalized relative to the wrist origin.")
    if os.path.exists(HAND_LANDMARKS_IMG):
        st.image(HAND_LANDMARKS_IMG, use_container_width=True)


# ==============================================================================
# 2. LIVE RECOGNITION
# ==============================================================================
elif nav_choice == "📷 Live Recognition":
    st.subheader("📷 Real-Time Sign Language Recognition")
    st.write("Recognize static ASL alphabet gestures in real time using your webcam or uploaded snapshots.")

    if not has_models:
        st.warning("⚠️ Trained models were not detected in `models/`. Please run `python train_models.py` before live recognition.")
        st.stop()

    predictor = get_predictor()

    col_ctrl, col_view = st.columns([1, 2])

    with col_ctrl:
        st.markdown("### ⚙️ Controls")
        selected_model = st.selectbox(
            "Select ML Classifier:",
            ["Random Forest", "SVM", "KNN", "Decision Tree"]
        )

        input_mode = st.radio("Input Source:", ["Live Webcam Stream", "Camera Snapshot", "Upload Image File"])

        st.info("💡 **Tips for best results:**\n- Hold your hand steady facing the camera.\n- Ensure adequate room lighting.\n- Keep your hand within camera view.")

    with col_view:
        if input_mode == "Live Webcam Stream":
            st.markdown("### Live Webcam Feed")
            run_live = st.toggle("Start Camera", value=False)
            frame_window = st.image([])
            status_placeholder = st.empty()

            if run_live:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not access webcam. Please verify camera permissions or try Camera Snapshot mode.")
                else:
                    while run_live:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to receive frame from webcam.")
                            break

                        # Mirror frame for intuitive user interaction
                        frame = cv2.flip(frame, 1)

                        # Predict frame
                        pred_res = predictor.predict_frame(frame, model_name=selected_model)

                        # Display annotated frame
                        annotated_rgb = cv2.cvtColor(pred_res["annotated_frame"], cv2.COLOR_BGR2RGB)
                        frame_window.image(annotated_rgb, channels="RGB", use_container_width=True)

                        # Update status badge
                        if pred_res["hand_detected"]:
                            conf_str = f" ({pred_res['confidence']*100:.1f}% confidence)" if pred_res["confidence"] else ""
                            status_placeholder.markdown(f"""
                            <div class="prediction-badge">
                                <div class="metric-label">Predicted Letter</div>
                                <div class="predicted-letter">{pred_res['predicted_letter']}</div>
                                <div class="status-text">{pred_res['status']}{conf_str}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            status_placeholder.markdown("""
                            <div class="prediction-badge" style="border-color: #282b34; background-color: #181a20;">
                                <div class="status-text" style="color: #6b7280;">✋ No hand detected in frame</div>
                            </div>
                            """, unsafe_allow_html=True)

                    cap.release()

        elif input_mode == "Camera Snapshot":
            camera_image = st.camera_input("Take a photo of an ASL alphabet hand gesture")
            if camera_image is not None:
                bytes_data = camera_image.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                pred_res = predictor.predict_frame(cv_img, model_name=selected_model)
                annotated_rgb = cv2.cvtColor(pred_res["annotated_frame"], cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, caption="Annotated Detection", use_container_width=True)

                if pred_res["hand_detected"]:
                    conf_str = f" ({pred_res['confidence']*100:.1f}%)" if pred_res["confidence"] else ""
                    st.success(f"### Predicted Sign: **{pred_res['predicted_letter']}**{conf_str}")
                else:
                    st.warning("No hand landmarks detected. Ensure your hand is clearly visible.")

        elif input_mode == "Upload Image File":
            uploaded_file = st.file_uploader("Upload an image containing an ASL hand gesture", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                pred_res = predictor.predict_frame(cv_img, model_name=selected_model)
                annotated_rgb = cv2.cvtColor(pred_res["annotated_frame"], cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb, caption="Processed Image", use_container_width=True)

                if pred_res["hand_detected"]:
                    conf_str = f" ({pred_res['confidence']*100:.1f}%)" if pred_res["confidence"] else ""
                    st.success(f"### Predicted Sign: **{pred_res['predicted_letter']}**{conf_str}")
                else:
                    st.warning("No hand landmarks detected in uploaded image.")


# ==============================================================================
# 3. WORDS RECOGNITION (FINGERSPELLING & COMPLETE DYNAMIC SIGNS)
# ==============================================================================
elif nav_choice == "🔤 Words Recognition":
    st.subheader("🔤 Words & Sentence Recognition")
    st.write("Construct words through alphabet fingerspelling or recognize movement-based full signs using WLASL models.")

    rec_mode = st.radio(
        "Select Word Recognition Method:",
        ["🔤 Fingerspelling Word Builder", "🎬 Dynamic Sign-to-Word (WLASL)"],
        horizontal=True,
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # FEATURE A: FINGERSPELLING WORD BUILDER
    # --------------------------------------------------------------------------
    if rec_mode == "🔤 Fingerspelling Word Builder":
        if not has_models:
            st.warning("⚠️ Trained alphabet models were not detected in `models/`. Please run `python train_models.py` first.")
            st.stop()

        from src.fingerspelling import FingerspellingManager
        if "fs_manager" not in st.session_state:
            st.session_state["fs_manager"] = FingerspellingManager()

        fs = st.session_state["fs_manager"]
        predictor = get_predictor()

        col_left, col_right = st.columns([1, 1.3])

        with col_left:
            st.markdown("### ⚙️ Fingerspelling Controls")
            fs_model = st.selectbox(
                "Alphabet Classifier:",
                ["Random Forest", "SVM", "KNN", "Decision Tree"],
                key="fs_model_select"
            )

            fs_input = st.radio(
                "Input Source:",
                ["Live Webcam Stream", "Camera Snapshot", "Upload Image File"],
                key="fs_input_radio"
            )

            with st.expander("🛠️ Debouncing & Stability Settings", expanded=False):
                fs.auto_commit = st.checkbox("Auto-commit letters when gesture held", value=fs.auto_commit)
                fs.min_hold_frames = st.slider("Hold frames threshold", 3, 20, fs.min_hold_frames)
                fs.min_confidence = st.slider("Minimum confidence", 0.20, 0.95, fs.min_confidence, 0.05)

            st.markdown("### 🔠 Word Actions")
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            with btn_c1:
                if st.button("➕ Confirm", use_container_width=True, help="Commit candidate letter (supports double letters like LL in HELLO)"):
                    fs.confirm_letter()
                    st.rerun()
            with btn_c2:
                if st.button("↩️ Undo", use_container_width=True, help="Remove last letter"):
                    fs.undo_letter()
                    st.rerun()
            with btn_c3:
                if st.button("␣ Space", use_container_width=True, help="Finish current word and start next"):
                    fs.add_space()
                    st.rerun()

            btn_c4, btn_c5 = st.columns(2)
            with btn_c4:
                if st.button("🗑️ Clear Word", use_container_width=True):
                    fs.clear_current_word()
                    st.rerun()
            with btn_c5:
                if st.button("🔄 Clear All", use_container_width=True):
                    fs.clear_all()
                    st.rerun()

        with col_right:
            # Word display cards
            curr_w = fs.current_word if fs.current_word else "—"
            full_t = fs.get_transcript() if fs.get_transcript() else "—"

            st.markdown(f"""
            <div style="background-color: #12141a; border: 1px solid #282b34; border-radius: 6px; padding: 14px 20px; margin-bottom: 12px;">
                <div style="font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Newsreader', serif;">Current Word</div>
                <div style="font-size: 34px; font-weight: 700; color: #ef4444; letter-spacing: 0.12em; font-family: 'Newsreader', serif; min-height: 44px; display: flex; align-items: center;">
                    {curr_w}
                </div>
            </div>
            <div style="background-color: #12141a; border: 1px solid #282b34; border-radius: 6px; padding: 12px 18px; margin-bottom: 16px;">
                <div style="font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Newsreader', serif;">Full Transcript</div>
                <div style="font-size: 18px; color: #f3f4f6; font-family: 'Newsreader', serif; min-height: 24px;">
                    {full_t}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if fs_input == "Live Webcam Stream":
                run_fs_live = st.toggle("Start Fingerspelling Camera", value=False, key="fs_cam_toggle")
                fs_frame_win = st.empty()
                fs_status_ph = st.empty()

                if run_fs_live:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ Could not access webcam. Please check camera permissions or use Camera Snapshot.")
                    else:
                        while run_fs_live:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.flip(frame, 1)
                            pred = predictor.predict_frame(frame, model_name=fs_model)
                            annotated_rgb = cv2.cvtColor(pred["annotated_frame"], cv2.COLOR_BGR2RGB)
                            fs_frame_win.image(annotated_rgb, channels="RGB", use_container_width=True)

                            if pred["hand_detected"]:
                                status = fs.process_frame_prediction(pred["predicted_letter"], pred["confidence"])
                                conf_pct = f"{pred['confidence']*100:.1f}%" if pred["confidence"] else "N/A"
                                lock_txt = "🟢 Ready" if status["gesture_released"] else "🔒 Release gesture for same letter"
                                fs_status_ph.markdown(f"""
                                <div class="prediction-badge" style="padding: 10px 14px; margin-top: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div><strong>Detected:</strong> <span style="color: #ef4444; font-size: 20px; font-weight: bold;">{pred['predicted_letter']}</span> ({conf_pct})</div>
                                        <div style="font-size: 13px; color: #9ca3af;">{lock_txt}</div>
                                    </div>
                                    <div style="margin-top: 6px; font-size: 12px; color: #9ca3af;">Hold progress: {int(status['hold_progress']*100)}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                fs.process_frame_prediction(None)
                                fs_status_ph.markdown("""
                                <div class="prediction-badge" style="border-color: #282b34; background-color: #181a20; padding: 10px 14px; margin-top: 8px;">
                                    <div style="color: #6b7280; font-size: 13px;">✋ Show an ASL hand gesture to spell</div>
                                </div>
                                """, unsafe_allow_html=True)

                        cap.release()

            elif fs_input == "Camera Snapshot":
                snap = st.camera_input("Capture an alphabet gesture for fingerspelling", key="fs_snap")
                if snap is not None:
                    bytes_data = snap.getvalue()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    pred = predictor.predict_frame(cv_img, model_name=fs_model)
                    st.image(cv2.cvtColor(pred["annotated_frame"], cv2.COLOR_BGR2RGB), use_container_width=True)

                    if pred["hand_detected"]:
                        letter = pred["predicted_letter"]
                        conf_str = f" ({pred['confidence']*100:.1f}%)" if pred["confidence"] else ""
                        st.success(f"Detected Letter: **{letter}**{conf_str}")
                        if st.button(f"Add '{letter}' to Word"):
                            fs.confirm_letter(letter)
                            st.rerun()
                    else:
                        st.warning("No hand landmarks detected. Ensure hand is visible.")

            elif fs_input == "Upload Image File":
                up_file = st.file_uploader("Upload hand gesture image", type=["jpg", "jpeg", "png"], key="fs_upload")
                if up_file is not None:
                    cv_img = cv2.imdecode(np.asarray(bytearray(up_file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                    pred = predictor.predict_frame(cv_img, model_name=fs_model)
                    st.image(cv2.cvtColor(pred["annotated_frame"], cv2.COLOR_BGR2RGB), use_container_width=True)

                    if pred["hand_detected"]:
                        letter = pred["predicted_letter"]
                        conf_str = f" ({pred['confidence']*100:.1f}%)" if pred["confidence"] else ""
                        st.success(f"Detected Letter: **{letter}**{conf_str}")
                        if st.button(f"Add '{letter}' to Word", key="btn_add_uploaded"):
                            fs.confirm_letter(letter)
                            st.rerun()
                    else:
                        st.warning("No hand landmarks detected in uploaded image.")

    # --------------------------------------------------------------------------
    # FEATURE B: COMPLETE SIGN-TO-WORD (WLASL)
    # --------------------------------------------------------------------------
    elif rec_mode == "🎬 Dynamic Sign-to-Word (WLASL)":
        word_predictor = get_word_predictor()
        has_word_models_trained = word_predictor.is_available()

        if not has_word_models_trained:
            st.warning("⚠️ **Trained Word Models Not Found**")
            st.info("""
            Word-level recognition classifies complete dynamic gestures using the **WLASL (World-Level American Sign Language)** dataset.
            
            **To download the dataset and train classical ML classifiers (KNN, SVM, Decision Tree, Random Forest):**
            ```bash
            # 1. Download official WLASL video subset
            python download_word_dataset.py --glosses book drink help no yes --max_per_gloss 2

            # 2. Extract 315-D temporal MediaPipe landmark features
            python extract_word_landmarks.py

            # 3. Train all 4 classical ML pipelines
            python train_word_models.py
            ```
            """)
            st.stop()

        supported_words = word_predictor.get_supported_classes()
        st.markdown(f"**Recognized Vocabulary ({len(supported_words)} signs):** " + " ".join([f"`{w}`" for w in supported_words]))

        w_col_ctrl, w_col_view = st.columns([1, 1.5])

        with w_col_ctrl:
            st.markdown("### ⚙️ Sign Classifier")
            selected_word_model = st.selectbox(
                "Select Classifier:",
                ["Decision Tree", "Random Forest", "SVM", "KNN"],
                key="word_model_select"
            )

            word_input_mode = st.radio(
                "Capture Mode:",
                ["Live Camera Sign (2-Sec Gesture)", "Upload Sign Video Clip", "Word Models Benchmark"],
                key="word_input_mode_radio"
            )

        with w_col_view:
            if word_input_mode == "Live Camera Sign (2-Sec Gesture)":
                st.markdown("### 📹 Record Dynamic Gesture")
                st.caption("Click the button below to capture 40 frames (~1.5-2 seconds) of hand motion and recognize the complete sign.")

                if st.button("🔴 Record and Predict Sign", key="btn_record_sign"):
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ Could not access webcam.")
                    else:
                        frame_ph = st.empty()
                        prog_bar = st.progress(0.0)
                        captured_landmarks = []

                        total_record_frames = 40
                        for i in range(total_record_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.flip(frame, 1)
                            # Extract landmarks
                            res = word_predictor.extractor.extract_landmarks(frame)
                            if res is not None:
                                feats_63, raw_pts = res
                                captured_landmarks.append(feats_63)
                                annotated = word_predictor.extractor.draw_landmarks_on_image(frame, raw_pts)
                            else:
                                annotated = frame

                            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                            prog_bar.progress((i + 1) / total_record_frames)
                            import time
                            time.sleep(0.03)

                        cap.release()

                        # Run sequence prediction
                        st.markdown("---")
                        st.markdown("### 🔍 Prediction Result")
                        pred_output = word_predictor.predict_sequence(
                            captured_landmarks,
                            model_name=selected_word_model,
                            min_frames=5,
                            confidence_threshold=0.35,
                        )

                        if pred_output["predicted_word"]:
                            if pred_output["uncertain"]:
                                st.warning(f"⚠️ **{pred_output['status']}**")
                            else:
                                conf_str = f" ({pred_output['confidence']*100:.1f}%)" if pred_output["confidence"] else ""
                                st.markdown(f"""
                                <div class="prediction-badge" style="padding: 16px 20px;">
                                    <div class="metric-label">Predicted Complete Sign</div>
                                    <div class="predicted-letter" style="font-size: 32px;">{pred_output['predicted_word'].upper()}</div>
                                    <div class="status-text">{pred_output['status']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            if pred_output.get("top_candidates"):
                                st.markdown("#### Candidate Probabilities:")
                                for cand in pred_output["top_candidates"]:
                                    st.write(f"- **{cand['word']}**: {cand['confidence']*100:.1f}%")
                        else:
                            st.warning(f"⚠️ {pred_output['status']}")

            elif word_input_mode == "Upload Sign Video Clip":
                up_vid = st.file_uploader("Upload sign video (.mp4, .mov)", type=["mp4", "mov"], key="vid_upload")
                if up_vid is not None:
                    import tempfile
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(up_vid.read())
                    tfile.close()

                    from extract_word_landmarks import extract_video_landmarks
                    st.video(tfile.name)

                    with st.spinner("Extracting temporal MediaPipe landmarks..."):
                        seq = extract_video_landmarks(tfile.name, word_predictor.extractor)
                        try:
                            os.unlink(tfile.name)
                        except OSError:
                            pass

                    pred_output = word_predictor.predict_sequence(
                        seq,
                        model_name=selected_word_model,
                        min_frames=3,
                        confidence_threshold=0.35,
                    )

                    st.markdown("---")
                    st.markdown("### 🔍 Video Prediction Result")
                    if pred_output["predicted_word"]:
                        if pred_output["uncertain"]:
                            st.warning(f"⚠️ {pred_output['status']}")
                        else:
                            st.markdown(f"""
                            <div class="prediction-badge" style="padding: 16px 20px;">
                                <div class="metric-label">Predicted Complete Sign</div>
                                <div class="predicted-letter" style="font-size: 32px;">{pred_output['predicted_word'].upper()}</div>
                                <div class="status-text">{pred_output['status']} (analyzed {pred_output['frames_analyzed']} frames)</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {pred_output['status']}")

            elif word_input_mode == "Word Models Benchmark":
                st.markdown("### 📊 Word-Level Models Benchmark")
                w_comp = load_json(WORD_COMPARISON_FILE)
                if w_comp and "models" in w_comp:
                    w_rows = []
                    for mname, mdata in w_comp["models"].items():
                        w_rows.append({
                            "Model Name": mname,
                            "Accuracy": f"{mdata['accuracy']*100:.1f}%",
                            "Macro F1": f"{mdata['macro_f1']*100:.1f}%",
                            "Macro Precision": f"{mdata['macro_precision']*100:.1f}%",
                            "Macro Recall": f"{mdata['macro_recall']*100:.1f}%",
                            "Test Samples": mdata["test_samples"],
                        })
                    st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)
                    st.info(f"🏆 Top Word Model: **{w_comp.get('best_model', 'N/A')}** ({w_comp.get('best_accuracy', 0)*100:.1f}% accuracy)")
                else:
                    st.info("No comparison results found in results/word_results/word_model_comparison.json.")


# ==============================================================================
# 4. MODEL COMPARISON
# ==============================================================================
elif nav_choice == "📊 Model Comparison":
    st.subheader("📊 Machine Learning Model Comparison")
    st.write("Rigorous comparison of the 4 tuned ML classifiers evaluated on the strictly held-out test set.")

    comp_data = load_json(MODEL_COMPARISON_FILE)

    if not comp_data:
        st.warning("⚠️ Model evaluation results not found in `results/model_comparison.json`. Please run `python train_models.py` first.")
        st.stop()

    st.success(f"🏆 **Top Performing Model:** **{comp_data['best_model']}** with **{comp_data['best_f1_macro']*100:.2f}%** Macro F1-Score and **{comp_data['best_accuracy']*100:.2f}%** Accuracy.")

    # Table of comparison
    models_list = comp_data.get("models", [])
    table_rows = []
    for m in models_list:
        table_rows.append({
            "Model Name": m["model_name"],
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Macro Precision": f"{m['precision_macro']*100:.2f}%",
            "Macro Recall": f"{m['recall_macro']*100:.2f}%",
            "Macro F1-Score": f"{m['f1_macro']*100:.2f}%",
            "CV Best Score": f"{m.get('cv_best_score', 0)*100:.2f}%" if "cv_best_score" in m else "N/A"
        })

    df_comp = pd.DataFrame(table_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    col_chart, col_cm = st.columns([1, 1])

    with col_chart:
        st.markdown("### 📈 Accuracy & F1-Score Benchmark")
        if os.path.exists(COMPARISON_CHART_FILE):
            st.image(COMPARISON_CHART_FILE, use_container_width=True)
        else:
            st.info("Comparison chart not generated yet.")

    with col_cm:
        st.markdown("### 🔲 Confusion Matrix")
        model_names = [m["model_name"] for m in models_list]
        selected_cm_model = st.selectbox("Select Model for Confusion Matrix:", model_names)

        safe_slug = selected_cm_model.lower().replace(" ", "_")
        cm_file = os.path.join("results", f"confusion_matrix_{safe_slug}.png")
        if os.path.exists(cm_file):
            st.image(cm_file, use_container_width=True)
        else:
            st.info(f"Confusion matrix image for {selected_cm_model} not found.")

    st.markdown("---")
    st.markdown("### 📋 Per-Class Performance Breakdown")
    detail_model = st.selectbox("View Classification Report for:", model_names, key="detail_model_select")
    target_eval = next((m for m in models_list if m["model_name"] == detail_model), None)

    if target_eval and "classification_report" in target_eval:
        rep = target_eval["classification_report"]
        rows = []
        for k, v in rep.items():
            if isinstance(v, dict):
                rows.append({
                    "Class": k,
                    "Precision": f"{v['precision']:.3f}",
                    "Recall": f"{v['recall']:.3f}",
                    "F1-Score": f"{v['f1-score']:.3f}",
                    "Support": int(v['support'])
                })
        df_rep = pd.DataFrame(rows)
        st.dataframe(df_rep, use_container_width=True, hide_index=True)


# ==============================================================================
# 4. DATASET EXPLORER
# ==============================================================================
elif nav_choice == "📁 Dataset Explorer":
    st.subheader("📁 ASL Dataset Inspection & Cleaning Report")

    ds_summary = load_json(DATASET_SUMMARY_FILE)
    ext_summary = load_json(EXTRACTION_SUMMARY_FILE)

    if not ds_summary and not ext_summary:
        st.info("ℹ️ Dataset has not been inspected yet. Run `python download_dataset.py` followed by `python extract_landmarks.py`.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        valid_count = ds_summary.get("total_valid_images", "N/A") if ds_summary else "N/A"
        st.metric("Total Valid Images", valid_count)
    with col2:
        dup_count = ds_summary.get("total_duplicates", 0) if ds_summary else 0
        st.metric("Duplicates Filtered", dup_count)
    with col3:
        det_rate = f"{ext_summary.get('overall_detection_rate', 0)*100:.1f}%" if ext_summary else "N/A"
        st.metric("MediaPipe Hand Detection Rate", det_rate)

    st.markdown("---")
    st.markdown("### 📊 Class Distribution")

    if ext_summary and "class_stats" in ext_summary:
        stats = ext_summary["class_stats"]
        chart_data = pd.DataFrame([
            {"Class": k, "Detected Hands": v["detected"], "Rejected": v["rejected"]}
            for k, v in sorted(stats.items())
        ]).set_index("Class")
        st.bar_chart(chart_data)

    elif ds_summary and "class_counts" in ds_summary:
        counts = ds_summary["class_counts"]
        chart_data = pd.DataFrame(list(counts.items()), columns=["Class", "Image Count"]).set_index("Class")
        st.bar_chart(chart_data)

    if os.path.exists(LANDMARKS_FILE):
        st.markdown("---")
        st.markdown("### 📑 Landmarks Dataset Preview (`data/landmarks.csv`)")
        df_preview = pd.read_csv(LANDMARKS_FILE, nrows=10)
        st.dataframe(df_preview, use_container_width=True)


# ==============================================================================
# 5. ABOUT PROJECT
# ==============================================================================
elif nav_choice == "ℹ️ About Project":
    st.subheader("ℹ️ Project Architecture & Viva Guide")

    st.markdown("""
    ### 📌 Machine Learning Lab Mini-Project
    **Title:** Real-Time Sign Language Recognition Using MediaPipe and Machine Learning  
    **Subject:** Machine Learning Laboratory  

    ---

    ### 🧠 Technical Foundations

    #### 1. Why Landmark Extraction Instead of Raw Pixels?
    - Raw image pixels contain extraneous background information, skin tone variations, and high dimensionality ($200 \\times 200 \\times 3 = 120,000$ inputs).
    - MediaPipe extracts **21 precise 3D hand keypoints** ($63$ features).
    - Relative normalization ensures **translation invariance** (wrist origin at $0,0,0$) and **scale invariance** (normalized by hand magnitude), allowing the model to perform reliably irrespective of distance to camera.

    #### 2. The Four Machine Learning Classifiers
    1. **K-Nearest Neighbors (KNN):**
       - Distance-based instance learning using Euclidean / Manhattan distance.
       - Highly sensitive to feature magnitudes, hence paired with `StandardScaler` inside a Scikit-learn Pipeline.
    2. **Support Vector Machine (SVM):**
       - Projects the 63 normalized coordinates into higher-dimensional space using RBF and Linear kernels to find the maximum-margin hyperplane.
    3. **Decision Tree:**
       - Non-parametric recursive splitting based on Gini Impurity and Information Gain Entropy.
       - Inherently scale-invariant and interpretable.
    4. **Random Forest:**
       - Ensemble of decorrelated decision trees using bootstrap aggregation (bagging) and feature sub-sampling.
       - Highly resistant to overfitting and variance.

    #### 3. Data Leakage Prevention
    - Preprocessing scaling (`StandardScaler`) is strictly fit **only** on the training partition.
    - Test partition (20%) is completely held out until final evaluation.
    - Cross-validation during hyperparameter tuning is executed strictly within the training set.

    #### 4. Scope and Limitations
    - Recognizes **static alphabet signs** (A–Z).
    - Does not translate dynamic sign language gestures or continuous full sentences (which require temporal sequence models like LSTMs or Transformers).
    """)

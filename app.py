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

# Custom Styling: Restrained Computer Vision Research Lab Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography: Warm Charcoal */
    html, body, [class*="css"], .stMarkdown, .stText, p, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        color: #292524;
    }

    /* Restrained Headings Hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.02em;
        font-weight: 650;
        color: #1c1917;
    }

    /* Warm Off-White Research Canvas */
    [data-testid="stAppViewContainer"] {
        background-color: #faf8f5 !important;
    }

    /* Main Container */
    .main {
        background-color: transparent !important;
    }

    /* Top Header Bar */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Grounded Research Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f5f3ef !important;
        border-right: 1px solid #e7e5e4;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #f5f3ef !important;
    }

    /* Clean Research Lab Header */
    .header-card {
        background-color: #ffffff;
        border: 1px solid #e7e5e4;
        border-left: 3px solid #c2410c;
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }

    .header-card h1 {
        color: #1c1917 !important;
        font-size: 1.5rem;
        font-weight: 650;
        letter-spacing: -0.025em;
        margin-bottom: 4px;
    }

    .header-card p {
        color: #78716c;
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 0px;
    }

    /* Simple Metric Layout */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 6px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        text-align: left;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 650;
        color: #1c1917;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.8125rem;
        color: #78716c;
        font-weight: 500;
        text-transform: none;
        letter-spacing: normal;
        margin-top: 4px;
    }

    /* Clean Prediction Status Panel */
    .prediction-badge {
        background-color: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 6px;
        padding: 20px;
        text-align: center;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }

    .predicted-letter {
        font-size: 3.25rem;
        font-weight: 700;
        color: #c2410c;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .status-text {
        font-size: 0.9rem;
        font-weight: 500;
        color: #57534e;
        margin-top: 6px;
    }

    /* Subtle Dividers */
    hr {
        border-color: #e7e5e4 !important;
        margin: 20px 0 !important;
    }

    /* Native Research Lab Buttons */
    .stButton > button {
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        border: 1px solid #d6d3d1 !important;
        color: #292524 !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        border-color: #c2410c !important;
        color: #c2410c !important;
        background-color: #fff7ed !important;
    }

    /* Code Blocks & Terminal Boxes: High Contrast */
    pre, code, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
        background-color: #1c1917 !important;
        border: 1px solid #292524 !important;
        border-radius: 6px !important;
    }

    pre *, code *, [data-testid="stCodeBlock"] * {
        color: #f5f5f4 !important;
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
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


@st.cache_resource
def get_predictor():
    """Initializes and caches the sign language predictor."""
    from src.prediction import SignLanguagePredictor
    return SignLanguagePredictor()


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
    ["🏠 Home Dashboard", "📷 Live Recognition", "📊 Model Comparison", "📁 Dataset Explorer", "ℹ️ About Project"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

has_downloaded = os.path.exists(DATASET_INFO_FILE)
has_landmarks = os.path.exists(LANDMARKS_FILE)
has_models = os.path.exists(MODEL_METADATA_FILE)

st.sidebar.write(f"Dataset Download: {'✅ Ready' if has_downloaded else '⏳ Pending'}")
st.sidebar.write(f"Landmarks Extracted: {'✅ Ready' if has_landmarks else '⏳ Pending'}")
st.sidebar.write(f"Models Trained: {'✅ Ready' if has_models else '⏳ Pending'}")

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
    st.subheader("🚀 Quick Start / Reproduction Commands")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Kaggle dataset via KaggleHub
python download_dataset.py

# 3. Extract MediaPipe hand landmarks
python extract_landmarks.py

# 4. Train and tune all 4 ML models
python train_models.py

# 5. Launch interactive web dashboard
streamlit run app.py
    """, language="bash")


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
                            <div class="prediction-badge" style="border-color: #cbd5e1; background-color: #f8fafc;">
                                <div class="status-text" style="color: #94a3b8;">✋ No hand detected in frame</div>
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
# 3. MODEL COMPARISON
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

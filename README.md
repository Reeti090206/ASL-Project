# Real-Time Sign Language Recognition Using MediaPipe and Machine Learning

An end-to-end Machine Learning Laboratory mini-project for real-time American Sign Language (ASL) static alphabet recognition using MediaPipe 3D hand landmark extraction and Scikit-learn classical classifiers.

---

## 📌 Project Overview & Objectives

In this project, we implement a robust computer vision and machine learning pipeline that classifies static hand gestures representing letters of the American Sign Language alphabet.

### ML Lab Requirements Satisfied:
1. **Clean Custom Dataset & Prevent Data Leakage:** Stratified train/test splitting with data transformations fit strictly on training partitions using Scikit-learn Pipelines.
2. **Train and Tune 4 Classical Algorithms:** Rigorous comparison of strictly four required models:
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Machine (SVM)**
   - **Decision Tree**
   - **Random Forest**
   *(No deep learning or CNN classifiers used).*
3. **Interactive Streamlit Web Dashboard:** Comprehensive user interface featuring:
   - Real-time live webcam sign recognition with landmark skeleton overlay.
   - Model comparison tables, confusion matrices, and macro-averaged metrics.
   - Dataset explorer and validation report.
   - Viva & project explanation guide.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Purpose |
|---|---|---|
| **Programming Language** | Python 3.10 – 3.13 | Core runtime |
| **Dataset Download** | KaggleHub | Automatic zero-configuration dataset retrieval |
| **Computer Vision** | OpenCV (`cv2`) | Video frame capture and image manipulation |
| **Feature Extraction** | MediaPipe (`HandLandmarker`) | 21 3D hand keypoints ($63$ spatial features) |
| **Data Processing** | Pandas & NumPy | Data cleaning, validation, and array manipulation |
| **Machine Learning** | Scikit-learn | Pipelines, scaling, tuning (GridSearchCV), evaluation |
| **Model Serialization** | Joblib | High-efficiency pipeline serialization |
| **Visualizations** | Matplotlib | High-resolution confusion matrices and bar charts |
| **Web Interface** | Streamlit | Responsive, student-friendly interactive dashboard |
| **Testing** | Pytest | Automated non-webcam unit and integration tests |

---

## 📁 Repository Structure

```text
sign-language-recognition/
│
├── app.py                     # Interactive Streamlit Web Application
├── download_dataset.py        # Automatic KaggleHub dataset fetcher
├── extract_landmarks.py       # MediaPipe feature extraction (creates landmarks.csv)
├── train_models.py            # Pipelines, hyperparameter tuning & evaluation
├── requirements.txt           # Python dependency specification
├── README.md                  # Project documentation & execution guide
│
├── data/
│   ├── dataset_info.json      # Dataset path cache
│   └── landmarks.csv          # Extracted 63-D landmark features & labels
│
├── models/
│   ├── knn_model.joblib           # Trained KNN pipeline
│   ├── svm_model.joblib           # Trained SVM pipeline
│   ├── decision_tree_model.joblib # Trained Decision Tree pipeline
│   ├── random_forest_model.joblib # Trained Random Forest pipeline
│   └── model_metadata.json        # Class labels & model metadata
│
├── results/
│   ├── dataset_summary.json       # Dataset validation report
│   ├── extraction_summary.json    # Landmark extraction statistics
│   ├── model_comparison.json      # Comparative metrics (Acc, F1, Prec, Rec)
│   ├── model_comparison_chart.png # Accuracy & Macro F1 comparison chart
│   └── confusion_matrix_*.png     # Confusion matrix plots per model
│
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py       # Dataset inspection, duplicate & corruption checking
│   ├── feature_extraction.py  # MediaPipe unified extractor & relative normalization
│   ├── preprocessing.py       # Dataset loading, validation & stratified split
│   ├── evaluation.py          # Metrics computation and visualization plotting
│   └── prediction.py          # Real-time frame inference & model caching
│
└── tests/
    ├── test_dataset_utils.py       # File hashing & directory structure tests
    ├── test_feature_extraction.py  # Landmark normalization & invariance tests
    ├── test_preprocessing.py       # Data validation & leakage prevention tests
    ├── test_models.py              # 4 pipelines, tuning & evaluation tests
    └── test_prediction.py          # Prediction & frame inference tests
```

---

## 🚀 Setup & Execution Guide (Windows)

### 1. Clone or Open Workspace
Open PowerShell in the project directory:
```powershell
cd d:\PROJECTS\ASL
```

### 2. Set Up Virtual Environment (Recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Step-by-Step Pipeline Execution

#### Step A: Download Dataset via KaggleHub
```powershell
python download_dataset.py
```
*Downloads the dataset automatically using `kagglehub` and caches its path dynamically without hardcoding paths.*

#### Step B: Extract MediaPipe Hand Landmarks
```powershell
python extract_landmarks.py --samples-per-class 150
```
- Extracts 21 landmarks per valid hand image (63 normalized numerical features).
- Applies translation invariance (wrist at origin) and scale invariance (normalized by hand magnitude).
- Filters unreadable files and skips frames where no hand is detected.
- Saves clean features and labels to `data/landmarks.csv`.

#### Step C: Train, Tune & Evaluate the 4 Models
```powershell
python train_models.py
```
- Performs stratified splitting (80% train, 20% held-out test).
- Preprocessing (`StandardScaler`) is fit strictly on training data inside Scikit-learn Pipelines.
- Performs cross-validation hyperparameter tuning for KNN, SVM, Decision Tree, and Random Forest.
- Evaluates on the held-out test set (Accuracy, Macro Precision, Macro Recall, Macro F1-score).
- Saves serialized models to `models/` and charts/matrices to `results/`.

#### Step D: Run Automated Tests
```powershell
pytest -v
```
*Executes all unit and integration tests without requiring a physical webcam.*

#### Step E: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*Opens the web dashboard in your default browser at `http://localhost:8501`.*


**Answer:** Macro-averaging computes metrics independently for each class and averages them equally, treating rare and frequent alphabet classes with equal weight. This ensures the model does not achieve artificially high scores by performing well only on dominant classes.

### Q4: Which model performed best and why?
**Answer:** Support Vector Machines (SVM with RBF kernel) and Random Forests typically perform best on normalized landmark geometry. SVM creates maximum-margin separating hyperplanes in 63-dimensional continuous space, while Random Forest handles non-linear boundaries through decorrelated decision trees.

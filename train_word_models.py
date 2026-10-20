"""
Model training, evaluation, and serialization for Word-Level ASL Recognition.
Trains strictly four classical ML classifiers:
1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
Persists models to models/word_models/ and metrics to results/word_results/.
Zero data leakage: scaling fit strictly on training partitions via Pipelines.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.word_recognition import TEMPORAL_FEATURE_NAMES, WORD_MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = os.path.join("data", "wlasl", "word_landmarks.csv")
DEFAULT_RESULTS_DIR = os.path.join("results", "word_results")
COMPARISON_JSON_PATH = os.path.join(DEFAULT_RESULTS_DIR, "word_model_comparison.json")
METADATA_PATH = os.path.join(WORD_MODELS_DIR, "word_model_metadata.json")


def load_word_dataset(csv_path: str = DEFAULT_CSV_PATH) -> Tuple[np.ndarray, np.ndarray, list]:
    """Loads temporal features and labels from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}. Run extract_word_landmarks.py first.")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a 'label' column.")

    feature_cols = [c for c in df.columns if c not in ["label", "video_id"]]
    X = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=np.float32))
    y = np.array(df["label"].tolist(), dtype=str)
    classes = sorted(list(str(c) for c in np.unique(y)))

    return X, y, classes


def get_word_pipelines() -> Dict[str, Pipeline]:
    """Builds scikit-learn pipelines for the 4 classical models."""
    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=min(3, 5), metric="minkowski", weights="distance"))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=5.0, probability=True, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=12, min_samples_split=2, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
        ])
    }


def train_and_evaluate_word_models(
    csv_path: str = DEFAULT_CSV_PATH,
    models_dir: str = WORD_MODELS_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes training and evaluation of all 4 word-level models.
    """
    logger.info(f"Loading word dataset from {csv_path}...")
    X, y, classes = load_word_dataset(csv_path)
    logger.info(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features, {len(classes)} classes: {classes}")

    if len(classes) < 2:
        raise ValueError(f"Need at least 2 distinct word classes to train classifiers, found: {classes}")

    # Check if stratified split is possible (at least 2 samples per class)
    unique_labels, counts = np.unique(y, return_counts=True)
    min_class_count = np.min(counts)

    can_stratify = min_class_count >= 2 and len(X) >= 2 * len(classes)
    if can_stratify:
        # Ensure test set has at least len(classes) samples for stratification
        min_test_samples = len(classes)
        calculated_test_samples = max(min_test_samples, int(round(len(X) * test_size)))
        if calculated_test_samples >= len(X):
            calculated_test_samples = len(classes)
        actual_test_size = calculated_test_samples / len(X)
        stratify_arg = y
    else:
        actual_test_size = max(1, int(round(len(X) * test_size))) / len(X)
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, random_state=random_state, stratify=stratify_arg
    )

    logger.info(f"Train partition: {len(X_train)} samples | Test partition: {len(X_test)} samples")

    pipelines = get_word_pipelines()
    eval_results = {}
    saved_models = {}

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    filename_map = {
        "KNN": "knn_word_model.joblib",
        "SVM": "svm_word_model.joblib",
        "Decision Tree": "decision_tree_word_model.joblib",
        "Random Forest": "random_forest_word_model.joblib",
    }

    best_model_name = None
    best_acc = -1.0

    for name, pipe in pipelines.items():
        logger.info(f"Training word classifier: {name}...")
        # Adjust KNN neighbors if training set is small
        if name == "KNN":
            k_val = min(3, len(X_train) - 1 if len(X_train) > 1 else 1)
            pipe.set_params(classifier__n_neighbors=max(1, k_val))

        pipe.fit(X_train, y_train)

        # Predict test partition
        y_pred = pipe.predict(X_test)

        acc = float(round(accuracy_score(y_test, y_pred), 4))
        f1 = float(round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4))
        prec = float(round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4))
        rec = float(round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4))

        eval_results[name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "macro_precision": prec,
            "macro_recall": rec,
            "test_samples": int(len(X_test)),
            "train_samples": int(len(X_train)),
        }

        # Save model
        out_path = os.path.join(models_dir, filename_map[name])
        joblib.dump(pipe, out_path)
        saved_models[name] = out_path
        logger.info(f"Saved {name} -> {out_path} (Acc: {acc*100:.1f}%, F1: {f1*100:.1f}%)")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    comparison = {
        "models": eval_results,
        "best_model": best_model_name,
        "best_accuracy": best_acc,
        "classes": classes,
        "num_classes": len(classes),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_dim": X.shape[1],
    }

    # Save comparison and metadata
    comparison_file = os.path.join(results_dir, "word_model_comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    metadata = {
        "classes": classes,
        "num_classes": len(classes),
        "feature_dim": X.shape[1],
        "best_model": best_model_name,
        "best_accuracy": best_acc,
        "models": filename_map,
    }
    meta_file = os.path.join(models_dir, "word_model_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Word model training complete!")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train word-level ASL classical ML classifiers")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to word_landmarks.csv")
    parser.add_argument("--models_dir", default=WORD_MODELS_DIR, help="Output models directory")
    args = parser.parse_args()

    result = train_and_evaluate_word_models(csv_path=args.csv, models_dir=args.models_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

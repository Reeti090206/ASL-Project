"""
Training and hyperparameter tuning for the four strict ML models:
1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest

Employs Scikit-learn Pipelines, cross-validation tuning on training data only,
and evaluates on a completely held-out test set to guarantee zero data leakage.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.preprocessing import load_and_validate_dataset, split_dataset
from src.evaluation import evaluate_model, save_evaluation_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
RESULTS_DIR = "results"
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")


def build_model_pipelines() -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    """
    Defines Scikit-learn pipelines and hyperparameter tuning grids.
    Scales features strictly inside pipelines for distance-sensitive models (KNN, SVM).
    """
    pipelines_and_grids = {
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier())
            ]),
            {
                "classifier__n_neighbors": [3, 5, 7],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"]
            }
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=42))
            ]),
            {
                "classifier__C": [1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"]
            }
        ),
        "Decision Tree": (
            Pipeline([
                ("classifier", DecisionTreeClassifier(random_state=42))
            ]),
            {
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__criterion": ["gini", "entropy"]
            }
        ),
        "Random Forest": (
            Pipeline([
                ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [15, 25, None],
                "classifier__min_samples_split": [2, 5]
            }
        ),
    }
    return pipelines_and_grids


def train_and_tune_all(csv_path: str = os.path.join("data", "landmarks.csv")):
    """
    Orchestrates full dataset loading, train/test splitting, hyperparameter tuning via CV,
    serialization with joblib, and held-out test set evaluation.
    """
    logger.info("=" * 60)
    logger.info("STARTING ASL MODEL TRAINING & TUNING WORKFLOW")
    logger.info("=" * 60)

    # 1. Load and clean dataset
    df, X, y = load_and_validate_dataset(csv_path)
    classes = sorted(list(np.unique(y)))

    # 2. Stratified train/test split (Zero data leakage)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    # Determine CV folds based on minimum class size in training set
    _, train_counts = np.unique(y_train, return_counts=True)
    min_train_count = train_counts.min()
    cv_folds = max(2, min(5, min_train_count))
    logger.info(f"Using {cv_folds}-fold cross validation on training data.")

    pipelines_and_grids = build_model_pipelines()
    evaluations = []
    saved_model_paths = {}

    os.makedirs(MODELS_DIR, exist_ok=True)

    for model_name, (pipeline, param_grid) in pipelines_and_grids.items():
        logger.info("-" * 50)
        logger.info(f"Hyperparameter tuning & training: {model_name}...")

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        logger.info(f"[{model_name}] Best Parameters: {grid_search.best_params_}")
        logger.info(f"[{model_name}] Best CV Macro F1: {grid_search.best_score_:.4f}")

        # Save model pipeline
        file_slug = model_name.lower().replace(" ", "_")
        model_filename = f"{file_slug}_model.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(best_model, model_path)
        saved_model_paths[model_name] = model_path
        logger.info(f"Saved {model_name} model to {model_path}")

        # Evaluate on the completely held-out test set
        eval_result = evaluate_model(best_model, X_test, y_test, model_name, classes)
        eval_result["best_hyperparameters"] = grid_search.best_params_
        eval_result["cv_best_score"] = float(round(grid_search.best_score_, 4))
        evaluations.append(eval_result)

    # 3. Save comprehensive evaluation and visualization results
    logger.info("=" * 60)
    logger.info("EVALUATION & VISUALIZATION GENERATION")
    logger.info("=" * 60)
    summary_report = save_evaluation_results(evaluations, classes, RESULTS_DIR)

    # 4. Save metadata for application loading
    metadata = {
        "classes": classes,
        "num_classes": len(classes),
        "feature_dim": 63,
        "model_paths": saved_model_paths,
        "best_model": summary_report["best_model"],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "cv_folds": cv_folds,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved to {METADATA_FILE}")
    logger.info(f"TRAINING COMPLETE! Top Model: {summary_report['best_model']} "
                f"(F1-Score: {summary_report['best_f1_macro']}, Accuracy: {summary_report['best_accuracy']})")
    return summary_report, metadata


if __name__ == "__main__":
    train_and_tune_all()

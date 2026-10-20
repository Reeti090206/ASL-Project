"""
Model evaluation, comparison metrics calculation, confusion matrices, and visualization plots.
Computes Accuracy, Macro Precision, Macro Recall, and Macro F1-score without data leakage.
"""

import os
import json
import logging
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    classes: List[str]
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on the held-out test set.
    Returns metrics dictionary with accuracy, macro precision, recall, and f1.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

    results = {
        "model_name": model_name,
        "accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec_macro, 4)),
        "recall_macro": float(round(rec_macro, 4)),
        "f1_macro": float(round(f1_macro, 4)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    logger.info(
        f"[{model_name}] Test Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | "
        f"Macro Prec: {prec_macro:.4f} | Macro Rec: {rec_macro:.4f}"
    )
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    model_name: str,
    output_path: str
):
    """
    Generates and saves a clean confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    # Label values if reasonable number of classes
    if len(classes) <= 30:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val > 0:
                    plt.text(
                        j, i, format(val, "d"),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if val > thresh else "black",
                        fontsize=7
                    )

    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_model_comparison_chart(
    comparison_data: List[Dict[str, Any]],
    output_path: str = os.path.join("results", "model_comparison_chart.png")
):
    """
    Generates a grouped bar chart comparing Accuracy and Macro F1-score across all 4 models.
    """
    model_names = [d["model_name"] for d in comparison_data]
    accuracies = [d["accuracy"] * 100 for d in comparison_data]
    f1_scores = [d["f1_macro"] * 100 for d in comparison_data]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    rects1 = ax.bar(x - width/2, accuracies, width, label="Accuracy (%)", color="#1f77b4", edgecolor="#144a72")
    rects2 = ax.bar(x + width/2, f1_scores, width, label="Macro F1-Score (%)", color="#2ca02c", edgecolor="#1a5e1a")

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Machine Learning Model Performance Comparison (Held-Out Test Set)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="#f8f9fa")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Bar label values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved model comparison chart to {output_path}")


def save_evaluation_results(
    evaluations: List[Dict[str, Any]],
    classes: List[str],
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Saves comprehensive metrics, comparison JSON, per-model confusion matrix plots,
    and identifies the overall best model.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Identify best model by macro F1-score, then accuracy
    best_eval = max(evaluations, key=lambda x: (x["f1_macro"], x["accuracy"]))

    # Save confusion matrix plots
    for ev in evaluations:
        cm_array = np.array(ev["confusion_matrix"])
        safe_name = ev["model_name"].lower().replace(" ", "_")
        cm_path = os.path.join(results_dir, f"confusion_matrix_{safe_name}.png")
        plot_confusion_matrix(cm_array, classes, ev["model_name"], cm_path)

    # Save comparison bar chart
    comparison_chart_path = os.path.join(results_dir, "model_comparison_chart.png")
    plot_model_comparison_chart(evaluations, comparison_chart_path)

    # Save comparison summary JSON
    summary_report = {
        "best_model": best_eval["model_name"],
        "best_f1_macro": best_eval["f1_macro"],
        "best_accuracy": best_eval["accuracy"],
        "classes": classes,
        "models": evaluations
    }

    summary_file = os.path.join(results_dir, "model_comparison.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"All evaluation results stored in {results_dir}. Best model: {best_eval['model_name']}")
    return summary_report

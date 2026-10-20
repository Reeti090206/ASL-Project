import os
import tempfile
import joblib
import numpy as np
import pytest

from train_models import build_model_pipelines
from src.evaluation import evaluate_model, save_evaluation_results


def test_model_pipelines_and_training():
    pipelines_and_grids = build_model_pipelines()
    expected_models = {"KNN", "SVM", "Decision Tree", "Random Forest"}
    assert set(pipelines_and_grids.keys()) == expected_models

    # Create synthetic dataset with 3 classes
    np.random.seed(42)
    classes = ["A", "B", "C"]
    n_samples_per_class = 15
    X = np.random.randn(n_samples_per_class * 3, 63).astype(np.float32)
    y = np.array(["A"] * n_samples_per_class + ["B"] * n_samples_per_class + ["C"] * n_samples_per_class)

    X_train, X_test = X[:36], X[36:]
    y_train, y_test = y[:36], y[36:]

    with tempfile.TemporaryDirectory() as temp_results_dir:
        evaluations = []
        for model_name, (pipeline, _) in pipelines_and_grids.items():
            # Fit pipeline on training data
            pipeline.fit(X_train, y_train)

            # Evaluate on held-out test data
            res = evaluate_model(pipeline, X_test, y_test, model_name, classes)
            assert "accuracy" in res
            assert "f1_macro" in res
            assert "confusion_matrix" in res
            assert 0.0 <= res["accuracy"] <= 1.0
            assert 0.0 <= res["f1_macro"] <= 1.0
            evaluations.append(res)

            # Test Joblib serialization
            model_dump_path = os.path.join(temp_results_dir, f"{model_name}.joblib")
            joblib.dump(pipeline, model_dump_path)
            loaded_pipeline = joblib.load(model_dump_path)
            pred = loaded_pipeline.predict(X_test[:2])
            assert len(pred) == 2

        # Test evaluation saving and visualization generation
        summary = save_evaluation_results(evaluations, classes, results_dir=temp_results_dir)
        assert "best_model" in summary
        assert os.path.exists(os.path.join(temp_results_dir, "model_comparison.json"))
        assert os.path.exists(os.path.join(temp_results_dir, "model_comparison_chart.png"))

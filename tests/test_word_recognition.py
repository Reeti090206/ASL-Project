"""
Unit tests for Word-Level Recognition and Temporal Feature Extraction.
Tests feature extraction math, WordSignPredictor, model training logic, and uncertainty handling.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.word_recognition import (
    extract_temporal_features,
    WordSignPredictor,
    TEMPORAL_FEATURE_DIM,
)
from train_word_models import get_word_pipelines


def test_extract_temporal_features_dimensions():
    # Empty sequence
    empty_feat = extract_temporal_features(np.zeros((0, 63)))
    assert empty_feat.shape == (TEMPORAL_FEATURE_DIM,)
    assert np.all(empty_feat == 0)

    # 1-frame sequence
    single_frame = np.random.randn(1, 63).astype(np.float32)
    feat_single = extract_temporal_features(single_frame)
    assert feat_single.shape == (TEMPORAL_FEATURE_DIM,)

    # Multi-frame sequence (15 frames of 63 features)
    multi_frame = np.random.randn(15, 63).astype(np.float32)
    feat_multi = extract_temporal_features(multi_frame)
    assert feat_multi.shape == (TEMPORAL_FEATURE_DIM,)
    assert not np.any(np.isnan(feat_multi))


def test_word_pipelines_architecture():
    pipelines = get_word_pipelines()
    expected_models = ["KNN", "SVM", "Decision Tree", "Random Forest"]
    for m in expected_models:
        assert m in pipelines
        assert isinstance(pipelines[m], Pipeline)


def test_word_predictor_uncertainty_and_prediction():
    # Train mock pipelines on synthetic 2-class data
    X_syn = np.random.randn(20, TEMPORAL_FEATURE_DIM).astype(np.float32)
    y_syn = np.array(["book"] * 10 + ["drink"] * 10)

    temp_dir = tempfile.mkdtemp()
    try:
        import joblib
        import json

        pipelines = get_word_pipelines()
        for name, pipe in pipelines.items():
            pipe.fit(X_syn, y_syn)
            fn_map = {
                "KNN": "knn_word_model.joblib",
                "SVM": "svm_word_model.joblib",
                "Decision Tree": "decision_tree_word_model.joblib",
                "Random Forest": "random_forest_word_model.joblib",
            }
            joblib.dump(pipe, os.path.join(temp_dir, fn_map[name]))

        metadata = {
            "classes": ["book", "drink"],
            "num_classes": 2,
            "feature_dim": TEMPORAL_FEATURE_DIM,
        }
        with open(os.path.join(temp_dir, "word_model_metadata.json"), "w") as f:
            json.dump(metadata, f)

        predictor = WordSignPredictor(models_dir=temp_dir)
        assert predictor.is_available() is True
        assert predictor.get_supported_classes() == ["book", "drink"]

        # Test sequence prediction with 10 dummy frames
        dummy_seq = [list(np.random.randn(63)) for _ in range(10)]
        pred_res = predictor.predict_sequence(dummy_seq, model_name="Random Forest")

        assert pred_res["predicted_word"] in ["book", "drink"]
        assert pred_res["frames_analyzed"] == 10
        assert "confidence" in pred_res
        assert "uncertain" in pred_res

        # Test insufficient frames
        short_seq = [list(np.random.randn(63)) for _ in range(2)]
        short_res = predictor.predict_sequence(short_seq, min_frames=5)
        assert short_res["predicted_word"] is None
        assert "Insufficient motion" in short_res["status"]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

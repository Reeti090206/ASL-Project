import os
import tempfile
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
import pytest

from src.prediction import SignLanguagePredictor


def test_prediction_pipeline():
    with tempfile.TemporaryDirectory() as temp_models_dir:
        # Create a mock model and save as random_forest_model.joblib
        dummy = DummyClassifier(strategy="constant", constant="A")
        dummy.fit(np.zeros((10, 63)), ["A"] * 10)
        rf_path = os.path.join(temp_models_dir, "random_forest_model.joblib")
        joblib.dump(dummy, rf_path)

        predictor = SignLanguagePredictor(models_dir=temp_models_dir)

        # Test empty frame
        empty_res = predictor.predict_frame(None)
        assert empty_res["hand_detected"] is False
        assert empty_res["status"] == "Empty frame received"

        # Test black image frame (no hand should be detected by MediaPipe)
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        blank_res = predictor.predict_frame(blank_img, model_name="Random Forest")
        assert blank_res["hand_detected"] is False
        assert blank_res["predicted_letter"] is None

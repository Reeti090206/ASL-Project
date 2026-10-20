"""
Inference and prediction pipeline for single images and live webcam video frames.
Ensures identical landmark normalization, feature ordering, and pipeline transformations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np

from src.feature_extraction import HandLandmarkExtractor, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")


class SignLanguagePredictor:
    """
    Handles live inference with cached ML models and MediaPipe landmark extraction.
    """

    MODEL_FILENAMES = {
        "KNN": "knn_model.joblib",
        "SVM": "svm_model.joblib",
        "Decision Tree": "decision_tree_model.joblib",
        "Random Forest": "random_forest_model.joblib",
    }

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.extractor = HandLandmarkExtractor(min_detection_confidence=0.5)
        self.loaded_models: Dict[str, Any] = {}
        self.metadata: Optional[Dict[str, Any]] = None
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata file: {e}")

    def load_model(self, model_name: str):
        """Loads and caches a trained model pipeline to avoid per-frame disk I/O."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        filename = self.MODEL_FILENAMES.get(model_name)
        if not filename:
            raise ValueError(f"Unknown model name '{model_name}'. Available: {list(self.MODEL_FILENAMES.keys())}")

        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file for '{model_name}' not found at {path}. Please run train_models.py first."
            )

        logger.info(f"Loading {model_name} from {path} into cache...")
        model = joblib.load(path)
        self.loaded_models[model_name] = model
        return model

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        model_name: str = "Random Forest"
    ) -> Dict[str, Any]:
        """
        Processes a single BGR camera frame:
        1. MediaPipe detects hand landmarks.
        2. Normalizes features (translation & scale invariance).
        3. Runs selected trained model.
        4. Draws landmarks on frame for visual feedback.
        """
        result = {
            "hand_detected": False,
            "predicted_letter": None,
            "confidence": None,
            "status": "No hand detected",
            "annotated_frame": frame_bgr,
            "normalized_features": None
        }

        if frame_bgr is None or frame_bgr.size == 0:
            result["status"] = "Empty frame received"
            return result

        extraction_result = self.extractor.extract_landmarks(frame_bgr)
        if extraction_result is None:
            return result

        features_63, raw_coords = extraction_result
        result["hand_detected"] = True
        result["normalized_features"] = features_63

        # Draw landmarks onto the frame
        annotated_frame = self.extractor.draw_landmarks_on_image(frame_bgr, raw_coords)
        result["annotated_frame"] = annotated_frame

        # Load model and predict
        try:
            model = self.load_model(model_name)
            X_input = np.array([features_63], dtype=np.float32)

            pred_label = model.predict(X_input)[0]
            result["predicted_letter"] = str(pred_label)
            result["status"] = f"Detected: {pred_label}"

            # Calculate confidence if model supports probabilities or decision function
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_input)[0]
                    result["confidence"] = float(round(np.max(proba), 4))
                except Exception:
                    result["confidence"] = None
            else:
                result["confidence"] = None

        except Exception as e:
            result["status"] = f"Prediction Error: {str(e)}"
            logger.error(f"Prediction failed for model {model_name}: {e}")

        return result

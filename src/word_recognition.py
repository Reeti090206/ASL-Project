"""
Word-Level Sign Language Recognition Module.
Implements temporal feature extraction and inference for movement-based complete signs.
Compatible with classical ML classifiers (KNN, SVM, Decision Tree, Random Forest).
Completely isolated from alphabet models and pipelines.
"""

import os
import json
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import joblib
import numpy as np

from src.feature_extraction import HandLandmarkExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WORD_MODELS_DIR = os.path.join("models", "word_models")
WORD_METADATA_FILE = os.path.join(WORD_MODELS_DIR, "word_model_metadata.json")
TEMPORAL_FEATURE_DIM = 63 * 5  # mean, std, velocity, displacement, max_velocity = 315 features

TEMPORAL_FEATURE_NAMES = []
for stat in ["mean", "std", "vel", "disp", "maxvel"]:
    for i in range(63):
        TEMPORAL_FEATURE_NAMES.append(f"{stat}_{i}")


def extract_temporal_features(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Converts a variable-length sequence of 3D hand landmarks (T, 63)
    into a fixed-length 315-dimensional temporal feature vector for classical ML.

    Calculates:
    - Trajectory Mean (63): Average joint coordinates
    - Trajectory Std (63): Motion range/spread per joint
    - Average Velocity (63): Mean frame-to-frame delta
    - Net Displacement (63): Vector from first to last detected frame
    - Peak Velocity (63): Maximum joint speed across the gesture
    """
    if landmark_sequence is None or len(landmark_sequence) == 0:
        return np.zeros(TEMPORAL_FEATURE_DIM, dtype=np.float32)

    seq = np.asarray(landmark_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 63:
        raise ValueError(f"Expected landmark sequence of shape (T, 63), got {seq.shape}")

    # If only 1 frame was captured, duplicate to allow difference calculations
    if len(seq) == 1:
        seq = np.vstack([seq, seq])

    mean_vec = np.mean(seq, axis=0)
    std_vec = np.std(seq, axis=0)
    diffs = np.abs(np.diff(seq, axis=0))
    vel_vec = np.mean(diffs, axis=0)
    disp_vec = seq[-1] - seq[0]
    maxvel_vec = np.max(diffs, axis=0)

    temporal_vec = np.concatenate([mean_vec, std_vec, vel_vec, disp_vec, maxvel_vec])
    return temporal_vec.astype(np.float32)


class WordSignPredictor:
    """
    Handles live and clip-based word-level inference using trained classical ML models.
    Operates strictly from models/word_models/.
    """

    MODEL_FILENAMES = {
        "Random Forest": "random_forest_word_model.joblib",
        "SVM": "svm_word_model.joblib",
        "KNN": "knn_word_model.joblib",
        "Decision Tree": "decision_tree_word_model.joblib",
    }

    def __init__(self, models_dir: str = WORD_MODELS_DIR, buffer_size: int = 45):
        self.models_dir = models_dir
        self.buffer_size = buffer_size
        self.extractor = HandLandmarkExtractor(min_detection_confidence=0.5)
        self.loaded_models: Dict[str, Any] = {}
        self.metadata: Optional[Dict[str, Any]] = None
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self._load_metadata()

    def _load_metadata(self):
        meta_path = os.path.join(self.models_dir, "word_model_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load word model metadata: {e}")

    def is_available(self) -> bool:
        """Checks if any trained word-level model exists on disk."""
        if not os.path.exists(self.models_dir):
            return False
        for fn in self.MODEL_FILENAMES.values():
            if os.path.exists(os.path.join(self.models_dir, fn)):
                return True
        return False

    def get_supported_classes(self) -> List[str]:
        """Returns the list of word classes recognized by the models."""
        if self.metadata and "classes" in self.metadata:
            return list(self.metadata["classes"])
        return []

    def load_model(self, model_name: str):
        """Loads and caches a trained word pipeline."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        filename = self.MODEL_FILENAMES.get(model_name)
        if not filename:
            raise ValueError(f"Unknown word model name '{model_name}'. Available: {list(self.MODEL_FILENAMES.keys())}")

        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Word model for '{model_name}' not found at {path}. Run train_word_models.py first."
            )

        logger.info(f"Loading word model {model_name} from {path}...")
        model = joblib.load(path)
        self.loaded_models[model_name] = model
        return model

    def add_frame(self, frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Adds a video frame to the rolling temporal buffer.
        Extracts 63 landmarks if hand is detected.
        Returns (features_63, raw_coords) or None.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        result = self.extractor.extract_landmarks(frame_bgr)
        if result is not None:
            features_63, raw_coords = result
            self.frame_buffer.append(features_63)
            return features_63, raw_coords
        return None

    def clear_buffer(self):
        """Clears the temporal frame buffer."""
        self.frame_buffer.clear()

    def predict_sequence(
        self,
        landmark_sequence: Optional[List[List[float]]] = None,
        model_name: str = "Random Forest",
        min_frames: int = 5,
        confidence_threshold: float = 0.40,
    ) -> Dict[str, Any]:
        """
        Infers word from landmark sequence (or from rolling buffer if not provided).
        """
        result = {
            "predicted_word": None,
            "confidence": None,
            "status": "No sign evaluated",
            "frames_analyzed": 0,
            "top_candidates": [],
            "uncertain": False,
        }

        # Use passed sequence or rolling buffer
        seq_data = landmark_sequence if landmark_sequence is not None else list(self.frame_buffer)
        if len(seq_data) < min_frames:
            result["status"] = f"Insufficient motion ({len(seq_data)}/{min_frames} frames). Perform the complete sign."
            result["frames_analyzed"] = len(seq_data)
            return result

        result["frames_analyzed"] = len(seq_data)

        try:
            model = self.load_model(model_name)
            features = extract_temporal_features(np.array(seq_data, dtype=np.float32))
            X = np.array([features], dtype=np.float32)

            pred_word = model.predict(X)[0]
            result["predicted_word"] = str(pred_word)

            # Extract probabilities if available
            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(X)[0]
                    classes = getattr(model, "classes_", [])
                    candidates = sorted(
                        [{"word": str(c), "confidence": float(round(p, 4))} for c, p in zip(classes, probas)],
                        key=lambda x: x["confidence"],
                        reverse=True
                    )
                    result["top_candidates"] = candidates[:3]
                    top_conf = candidates[0]["confidence"] if candidates else None
                    result["confidence"] = top_conf

                    if top_conf is not None and top_conf < confidence_threshold:
                        result["uncertain"] = True
                        result["status"] = f"Uncertain prediction: '{pred_word}' (conf: {top_conf*100:.1f}% < {confidence_threshold*100:.0f}%)"
                    else:
                        result["status"] = f"Recognized: {pred_word} ({top_conf*100:.1f}%)" if top_conf else f"Recognized: {pred_word}"
                except Exception:
                    result["status"] = f"Recognized: {pred_word}"
            else:
                result["status"] = f"Recognized: {pred_word}"

        except Exception as e:
            result["status"] = f"Prediction error: {str(e)}"
            logger.error(f"Word prediction error: {e}")

        return result

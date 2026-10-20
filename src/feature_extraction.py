"""
Feature extraction module using MediaPipe for hand landmark detection.
Supports both modern MediaPipe Tasks API (Python 3.13+) and legacy Solutions API.
Extracts 21 hand landmarks (63 features: x, y, z) with consistent relative normalization.
"""

import os
import urllib.request
import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
TASK_MODEL_PATH = os.path.join("models", "hand_landmarker.task")

FEATURE_COLUMNS = []
for i in range(21):
    FEATURE_COLUMNS.extend([f"x_{i}", f"y_{i}", f"z_{i}"])


def ensure_task_model_downloaded(model_path: str = TASK_MODEL_PATH) -> str:
    """Ensures that hand_landmarker.task model is downloaded for MediaPipe Tasks API."""
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logger.info(f"Downloading MediaPipe HandLandmarker task model to {model_path}...")
    try:
        urllib.request.urlretrieve(TASK_MODEL_URL, model_path)
        logger.info(f"Model downloaded successfully. Size: {os.path.getsize(model_path)} bytes")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download MediaPipe task model: {e}")
        raise e


class HandLandmarkExtractor:
    """
    Unified Hand Landmark Extractor that supports both MediaPipe Tasks API
    and MediaPipe Solutions API.
    """

    def __init__(self, min_detection_confidence: float = 0.5, num_hands: int = 1):
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.use_tasks_api = False
        self.detector = None
        self.mp_hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        self._init_mediapipe()

    def _init_mediapipe(self):
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            # Legacy API (e.g. MediaPipe <= 0.10.14 on Python <= 3.11)
            logger.info("Initializing MediaPipe using Solutions API...")
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self.num_hands,
                min_detection_confidence=self.min_detection_confidence,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.use_tasks_api = False
        else:
            # Modern Tasks API (MediaPipe Tasks on Python 3.12, 3.13, 3.14+)
            logger.info("Initializing MediaPipe using Tasks API (HandLandmarker)...")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = ensure_task_model_downloaded()
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True

    @staticmethod
    def normalize_landmarks(raw_coords: List[Tuple[float, float, float]]) -> List[float]:
        """
        Applies consistent translation and scale normalization to 21 landmarks.
        1. Translation invariance: Subtract wrist (landmark 0) from all landmarks.
        2. Scale invariance: Divide by maximum coordinate magnitude.
        Returns a 63-element 1D list [x0, y0, z0, ..., x20, y20, z20].
        """
        if len(raw_coords) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(raw_coords)}")

        origin_x, origin_y, origin_z = raw_coords[0]

        # Shift relative to wrist
        rel_coords = [
            (x - origin_x, y - origin_y, z - origin_z)
            for (x, y, z) in raw_coords
        ]

        # Maximum distance for scale normalization
        max_val = max(
            max(abs(x), abs(y), abs(z))
            for (x, y, z) in rel_coords
        )

        if max_val == 0.0:
            max_val = 1.0  # Avoid division by zero

        normalized = []
        for x, y, z in rel_coords:
            normalized.extend([x / max_val, y / max_val, z / max_val])

        return normalized

    def extract_landmarks(
        self, image_bgr: np.ndarray
    ) -> Optional[Tuple[List[float], List[Tuple[float, float, float]]]]:
        """
        Extracts 63 normalized landmark features from a BGR image.
        Returns:
            Tuple: (normalized_features_63, raw_landmarks_21)
            or None if no hand detected.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        import mediapipe as mp

        if self.use_tasks_api:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)

            if not detection_result.hand_landmarks:
                return None

            first_hand = detection_result.hand_landmarks[0]
            raw_coords = [(lm.x, lm.y, lm.z) for lm in first_hand]
            normalized_features = self.normalize_landmarks(raw_coords)
            return normalized_features, raw_coords

        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                return None

            first_hand = results.multi_hand_landmarks[0]
            raw_coords = [(lm.x, lm.y, lm.z) for lm in first_hand.landmark]
            normalized_features = self.normalize_landmarks(raw_coords)
            return normalized_features, raw_coords

    def draw_landmarks_on_image(
        self, image_bgr: np.ndarray, raw_coords: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Draws hand landmarks and skeleton connections on an image frame for visual feedback.
        Works across all platforms without depending on internal MediaPipe drawing modules.
        """
        annotated = image_bgr.copy()
        h, w, _ = annotated.shape

        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17)                                # Palm base
        ]

        # Draw connections
        pixel_points = []
        for x, y, z in raw_coords:
            px, py = int(x * w), int(y * h)
            pixel_points.append((px, py))

        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pixel_points[start_idx]
            pt2 = pixel_points[end_idx]
            cv2.line(annotated, pt1, pt2, (0, 200, 255), 2)  # Amber/Yellow bones

        # Draw landmark points
        for i, (px, py) in enumerate(pixel_points):
            radius = 5 if i in (4, 8, 12, 16, 20) else 3
            color = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (255, 128, 0)
            cv2.circle(annotated, (px, py), radius, color, -1)

        return annotated

import numpy as np
import pytest

from src.feature_extraction import HandLandmarkExtractor, FEATURE_COLUMNS


def test_feature_columns_dimension():
    assert len(FEATURE_COLUMNS) == 63
    assert FEATURE_COLUMNS[0] == "x_0"
    assert FEATURE_COLUMNS[1] == "y_0"
    assert FEATURE_COLUMNS[2] == "z_0"
    assert FEATURE_COLUMNS[62] == "z_20"


def test_normalize_landmarks_invariance():
    # Construct 21 landmarks
    raw_coords = [(float(i), float(i * 2), float(i * 3)) for i in range(21)]

    # Normalized features
    features = HandLandmarkExtractor.normalize_landmarks(raw_coords)
    assert len(features) == 63

    # Wrist landmark (index 0) shifted relative to itself must be (0.0, 0.0, 0.0)
    assert features[0] == 0.0
    assert features[1] == 0.0
    assert features[2] == 0.0

    # Values must be within [-1.0, 1.0] due to scale normalization
    for val in features:
        assert -1.0 <= val <= 1.0

    # Test translation invariance: shift all raw coordinates by (+100, +200, -50)
    shifted_coords = [(x + 100.0, y + 200.0, z - 50.0) for (x, y, z) in raw_coords]
    shifted_features = HandLandmarkExtractor.normalize_landmarks(shifted_coords)

    # Features must match identically despite translation
    np.testing.assert_allclose(features, shifted_features, rtol=1e-5, atol=1e-5)


def test_normalize_invalid_length():
    with pytest.raises(ValueError):
        HandLandmarkExtractor.normalize_landmarks([(0.1, 0.2, 0.3)] * 15)


def test_draw_landmarks():
    extractor = HandLandmarkExtractor(min_detection_confidence=0.5)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    coords = [(0.5, 0.5, 0.0)] * 21
    annotated = extractor.draw_landmarks_on_image(dummy_frame, coords)
    assert annotated.shape == (100, 100, 3)
    # Ensure drawing occurred (frame is no longer all zeroes)
    assert np.any(annotated > 0)

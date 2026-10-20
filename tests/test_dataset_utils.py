import os
import tempfile
import cv2
import numpy as np
import pytest

from src.dataset_utils import (
    compute_file_hash,
    find_dataset_image_dirs,
    validate_and_clean_dataset,
)


def test_compute_file_hash():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"hello ASL world")
        f_path = f.name
    try:
        hash1 = compute_file_hash(f_path)
        hash2 = compute_file_hash(f_path)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_find_and_validate_dataset_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock structure: root/train/A and root/train/B
        class_a_dir = os.path.join(temp_dir, "train", "A")
        class_b_dir = os.path.join(temp_dir, "train", "B")
        os.makedirs(class_a_dir)
        os.makedirs(class_b_dir)

        # Write dummy images
        img_a = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(class_a_dir, "a1.jpg"), img_a)
        cv2.imwrite(os.path.join(class_a_dir, "a2.jpg"), img_a)

        img_b = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.imwrite(os.path.join(class_b_dir, "b1.jpg"), img_b)

        # Write corrupt text file masquerading as image
        corrupt_path = os.path.join(class_b_dir, "corrupt.jpg")
        with open(corrupt_path, "w") as f:
            f.write("not an image")

        class_dict = find_dataset_image_dirs(temp_dir)
        assert "A" in class_dict
        assert "B" in class_dict
        assert len(class_dict["A"]) == 2
        assert len(class_dict["B"]) == 2

        summary, valid_dict = validate_and_clean_dataset(
            temp_dir, check_duplicates=True
        )
        assert summary["total_classes"] == 2
        assert summary["total_corrupted"] == 1  # The corrupt file was detected
        assert len(valid_dict["A"]) == 1  # Duplicate a2 was filtered from unique
        assert len(valid_dict["B"]) == 1  # Only b1 is valid

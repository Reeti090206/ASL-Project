"""
Dataset utilities for inspecting, validating, cleaning, and summarizing the ASL dataset.
"""

import os
import glob
import json
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def compute_file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Computes SHA-256 hash of a file for duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_dataset_image_dirs(root_path: str) -> Dict[str, List[str]]:
    """
    Recursively scans the root_path to find image class directories.
    Handles nested structures, e.g. root/asl_alphabet_train/asl_alphabet_train/A
    Returns a dictionary mapping class_label -> list of image filepaths.
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root dataset directory does not exist: {root_path}")

    class_images: Dict[str, List[str]] = {}

    IGNORE_DIR_NAMES = {"asl_alphabet_test", "test", "val", "validation", "eval"}

    # Walk through the directory tree looking for directories containing images
    for dirpath, dirnames, filenames in os.walk(root_path):
        folder_name = os.path.basename(dirpath).strip()
        if folder_name.lower() in IGNORE_DIR_NAMES:
            continue

        # Filter for image files
        imgs = [
            os.path.join(dirpath, f)
            for f in filenames
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if imgs:
            # Standardize label if single character or word (e.g. 'A', 'space', 'del', 'nothing')
            label = folder_name.upper() if len(folder_name) == 1 else folder_name
            if label not in class_images:
                class_images[label] = []
            class_images[label].extend(imgs)

    return class_images


def validate_and_clean_dataset(
    dataset_path: str,
    check_duplicates: bool = True,
    sample_validation_limit: Optional[int] = None,
) -> Dict:
    """
    Validates dataset files, verifies image readability, detects duplicates and corrupted files,
    and returns a comprehensive validation summary.
    Never modifies or deletes the original files.
    """
    logger.info(f"Validating dataset at: {dataset_path}")
    raw_class_dict = find_dataset_image_dirs(dataset_path)

    if not raw_class_dict:
        raise ValueError(f"No image files found in {dataset_path} with extensions {IMAGE_EXTENSIONS}")

    total_images_found = sum(len(imgs) for imgs in raw_class_dict.values())
    logger.info(f"Found {len(raw_class_dict)} classes with total {total_images_found} images.")

    class_counts = {}
    valid_images_by_class: Dict[str, List[str]] = {}
    corrupted_images: List[str] = []
    duplicate_images: List[str] = []
    seen_hashes = set()

    for class_label, img_paths in raw_class_dict.items():
        valid_images_by_class[class_label] = []
        paths_to_check = img_paths if sample_validation_limit is None else img_paths[:sample_validation_limit]

        for p in paths_to_check:
            # Check readability
            img = cv2.imread(p)
            if img is None:
                corrupted_images.append(p)
                continue

            # Duplicate check
            if check_duplicates:
                file_hash = compute_file_hash(p)
                if file_hash in seen_hashes:
                    duplicate_images.append(p)
                    # We still keep the original, duplicate is noted
                    continue
                seen_hashes.add(file_hash)

            valid_images_by_class[class_label].append(p)

        class_counts[class_label] = len(valid_images_by_class[class_label])

    # Summary
    classes_sorted = sorted(valid_images_by_class.keys())
    total_valid = sum(len(v) for v in valid_images_by_class.values())

    summary = {
        "dataset_path": dataset_path,
        "total_classes": len(classes_sorted),
        "classes": classes_sorted,
        "total_images_found": total_images_found,
        "total_valid_images": total_valid,
        "total_corrupted": len(corrupted_images),
        "total_duplicates": len(duplicate_images),
        "class_counts": class_counts,
        "is_sufficient": total_valid >= 50 and len(classes_sorted) >= 2,
    }

    # Save summary report
    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset validation completed. Summary saved to {summary_path}")
    return summary, valid_images_by_class

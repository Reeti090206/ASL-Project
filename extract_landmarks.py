"""
Script to extract MediaPipe 3D hand landmarks (63 features) from dataset images.
Saves normalized landmarks and labels to data/landmarks.csv.
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional
import pandas as pd
import cv2
from tqdm import tqdm

from src.dataset_utils import find_dataset_image_dirs, validate_and_clean_dataset
from src.feature_extraction import HandLandmarkExtractor, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_INFO_FILE = os.path.join("data", "dataset_info.json")
OUTPUT_CSV_FILE = os.path.join("data", "landmarks.csv")
EXTRACTION_SUMMARY_FILE = os.path.join("results", "extraction_summary.json")


def resolve_dataset_path(provided_path: Optional[str] = None) -> str:
    """Resolves the dataset directory path from CLI argument or cached dataset_info.json."""
    if provided_path and os.path.exists(provided_path):
        return provided_path

    if os.path.exists(DATASET_INFO_FILE):
        with open(DATASET_INFO_FILE, "r", encoding="utf-8") as f:
            info = json.load(f)
            path = info.get("dataset_path")
            if path and os.path.exists(path):
                return path

    # If neither exists, attempt to run download_dataset.py
    logger.info("Dataset path not specified or found. Attempting automatic download via KaggleHub...")
    from download_dataset import download_asl_dataset
    return download_asl_dataset()


def extract_landmarks_from_dataset(
    dataset_path: str,
    output_csv: str = OUTPUT_CSV_FILE,
    samples_per_class: Optional[int] = 150,
) -> pd.DataFrame:
    """
    Extracts 21 hand landmarks (63 features) per valid image and saves to CSV.
    Skips images where hand detection fails.
    """
    logger.info(f"Resolving images from dataset at: {dataset_path}")
    summary, valid_images_by_class = validate_and_clean_dataset(
        dataset_path,
        check_duplicates=True,
        sample_validation_limit=(samples_per_class * 2) if samples_per_class else None
    )

    extractor = HandLandmarkExtractor(min_detection_confidence=0.5)

    records = []
    total_images_processed = 0
    total_hands_detected = 0
    total_hands_rejected = 0
    class_stats = {}

    for label, img_paths in valid_images_by_class.items():
        # Select subset if samples_per_class specified
        selected_paths = img_paths[:samples_per_class] if samples_per_class else img_paths
        class_detected = 0
        class_rejected = 0

        logger.info(f"Extracting landmarks for class '{label}' ({len(selected_paths)} images)...")
        for img_path in tqdm(selected_paths, desc=f"Class {label}", leave=False):
            total_images_processed += 1
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                class_rejected += 1
                total_hands_rejected += 1
                continue

            extracted = extractor.extract_landmarks(img_bgr)
            if extracted is None:
                class_rejected += 1
                total_hands_rejected += 1
                continue

            features_63, _ = extracted
            # Validate feature dimensions
            if len(features_63) != 63:
                class_rejected += 1
                total_hands_rejected += 1
                continue

            row = [label] + features_63
            records.append(row)
            class_detected += 1
            total_hands_detected += 1

        class_stats[label] = {
            "processed": len(selected_paths),
            "detected": class_detected,
            "rejected": class_rejected,
            "detection_rate": round(class_detected / len(selected_paths), 4) if selected_paths else 0,
        }

    if not records:
        raise RuntimeError("No hand landmarks were detected in any of the dataset images!")

    # Build DataFrame
    columns = ["label"] + FEATURE_COLUMNS
    df = pd.DataFrame(records, columns=columns)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Extracted landmarks saved to: {output_csv} (Total samples: {len(df)})")

    # Save extraction summary
    os.makedirs("results", exist_ok=True)
    summary_data = {
        "dataset_path": dataset_path,
        "total_images_processed": total_images_processed,
        "total_hands_detected": total_hands_detected,
        "total_hands_rejected": total_hands_rejected,
        "overall_detection_rate": round(total_hands_detected / total_images_processed, 4) if total_images_processed else 0,
        "total_classes": len(class_stats),
        "classes": sorted(list(class_stats.keys())),
        "class_stats": class_stats,
    }

    with open(EXTRACTION_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Extraction summary report saved to: {EXTRACTION_SUMMARY_FILE}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks from ASL dataset.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to downloaded dataset root")
    parser.add_argument("--output-csv", type=str, default=OUTPUT_CSV_FILE, help="Path to save landmarks CSV")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=150,
        help="Max samples per class to extract (default: 150 for fast & high-accuracy training. Use 0 for all)",
    )
    args = parser.parse_args()

    limit = None if args.samples_per_class <= 0 else args.samples_per_class
    ds_path = resolve_dataset_path(args.dataset_path)
    extract_landmarks_from_dataset(ds_path, output_csv=args.output_csv, samples_per_class=limit)

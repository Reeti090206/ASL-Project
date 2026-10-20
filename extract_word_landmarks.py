"""
MediaPipe Temporal Landmark Extractor for Word-Level Video Sequences.
Processes video files, extracts per-frame hand landmarks, and computes
315-dimensional temporal feature vectors for movement-based signs.
Outputs data/wlasl/word_landmarks.csv and extraction summary.
"""

import os
import sys
import glob
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd

from src.feature_extraction import HandLandmarkExtractor
from src.word_recognition import extract_temporal_features, TEMPORAL_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VIDEOS_DIR = os.path.join("data", "wlasl", "videos")
DEFAULT_OUTPUT_CSV = os.path.join("data", "wlasl", "word_landmarks.csv")
DEFAULT_SUMMARY_JSON = os.path.join("results", "word_results", "word_extraction_summary.json")


def extract_video_landmarks(video_path: str, extractor: HandLandmarkExtractor, sample_fps: int = 15) -> List[List[float]]:
    """
    Reads a video file and extracts a list of 63-D normalized landmark vectors for frames where hand is detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Failed to open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / sample_fps)))

    sequence = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frame_interval == 0:
            result = extractor.extract_landmarks(frame)
            if result is not None:
                features_63, _ = result
                sequence.append(features_63)

        frame_idx += 1

    cap.release()
    return sequence


def process_word_videos(
    videos_dir: str = DEFAULT_VIDEOS_DIR,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    summary_path: str = DEFAULT_SUMMARY_JSON,
    min_sequence_length: int = 3,
) -> Dict[str, Any]:
    """
    Processes all video files in videos_dir, computes 315-D temporal features,
    and writes word_landmarks.csv.
    """
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4")) + glob.glob(os.path.join(videos_dir, "*.mov"))
    if not video_files:
        logger.error(f"No video files found in {videos_dir}. Please run download_word_dataset.py first.")
        return {"status": "error", "message": f"No video files found in {videos_dir}", "processed_samples": 0}

    logger.info(f"Initializing MediaPipe extractor for {len(video_files)} word sign videos...")
    extractor = HandLandmarkExtractor(min_detection_confidence=0.5)

    rows = []
    class_stats: Dict[str, int] = {}
    skipped_videos = 0

    for idx, vpath in enumerate(sorted(video_files), start=1):
        filename = os.path.basename(vpath)
        # Parse label from filename: e.g. "book_69241.mp4" -> label "book"
        label = filename.split("_")[0].lower() if "_" in filename else os.path.splitext(filename)[0].lower()

        logger.info(f"[{idx}/{len(video_files)}] Extracting landmarks for [{label}]: {filename}...")
        sequence = extract_video_landmarks(vpath, extractor)

        if len(sequence) < min_sequence_length:
            logger.warning(f"Skipping {filename}: only {len(sequence)} valid hand frames detected (< {min_sequence_length})")
            skipped_videos += 1
            continue

        temporal_vec = extract_temporal_features(np.array(sequence, dtype=np.float32))
        row_dict = {col: float(val) for col, val in zip(TEMPORAL_FEATURE_NAMES, temporal_vec)}
        row_dict["label"] = label
        row_dict["video_id"] = filename
        rows.append(row_dict)
        class_stats[label] = class_stats.get(label, 0) + 1

    if not rows:
        return {"status": "error", "message": "No valid hand landmark sequences could be extracted.", "processed_samples": 0}

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} temporal landmark samples across {len(class_stats)} classes to {output_csv}")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = {
        "status": "success",
        "total_videos_scanned": len(video_files),
        "total_samples_extracted": len(rows),
        "skipped_videos": skipped_videos,
        "classes": sorted(list(class_stats.keys())),
        "class_counts": class_stats,
        "feature_dim": len(TEMPORAL_FEATURE_NAMES),
        "output_csv": output_csv,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract temporal MediaPipe features from word sign videos")
    parser.add_argument("--videos_dir", default=DEFAULT_VIDEOS_DIR, help="Input videos directory")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    result = process_word_videos(videos_dir=args.videos_dir, output_csv=args.output_csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

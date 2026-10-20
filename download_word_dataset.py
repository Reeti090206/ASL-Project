"""
Automated downloader for WLASL (World-Level American Sign Language) dataset.
Fetches official WLASL metadata and downloads permitted real sign video files.
Isolates all downloads to data/wlasl/ to preserve existing datasets and models.
"""

import os
import sys
import json
import logging
import argparse
import urllib.request
from typing import List, Dict, Any, Optional
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
DEFAULT_DATA_DIR = os.path.join("data", "wlasl")
DEFAULT_VIDEOS_DIR = os.path.join(DEFAULT_DATA_DIR, "videos")
DEFAULT_METADATA_FILE = os.path.join(DEFAULT_DATA_DIR, "WLASL_v0.3.json")
DATASET_INFO_FILE = os.path.join(DEFAULT_DATA_DIR, "word_dataset_info.json")

# Core vocabulary of common signs available via direct permitted video hosting
DEFAULT_TARGET_GLOSSES = ["book", "drink", "help", "no", "yes", "go", "who", "all", "like", "mother"]


def download_wlasl_metadata(dest_path: str = DEFAULT_METADATA_FILE) -> Optional[List[Dict[str, Any]]]:
    """Downloads official WLASL_v0.3.json metadata if not already present."""
    if os.path.exists(dest_path):
        logger.info(f"WLASL metadata already present at: {dest_path}")
        try:
            with open(dest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing metadata: {e}. Re-downloading...")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading WLASL metadata from {WLASL_JSON_URL}...")
    try:
        req = urllib.request.Request(WLASL_JSON_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8")
            data = json.loads(content)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(content)
        logger.info(f"Successfully saved WLASL metadata ({len(data)} glosses) to {dest_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to download WLASL metadata: {e}")
        return None


def verify_video(video_path: str) -> bool:
    """Verifies that a video file is valid and readable by OpenCV."""
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret and frame is not None and frame.size > 0)
    except Exception:
        return False


def download_video_file(url: str, output_path: str, timeout: int = 15) -> bool:
    """Downloads a single video file via HTTP with user-agent and timeout."""
    if os.path.exists(output_path) and verify_video(output_path):
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, open(output_path, "wb") as out_file:
            out_file.write(response.read())

        if verify_video(output_path):
            return True
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    except Exception as e:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False


def download_wlasl_subset(
    target_glosses: List[str] = DEFAULT_TARGET_GLOSSES,
    max_instances_per_gloss: int = 4,
    data_dir: str = DEFAULT_DATA_DIR,
    videos_dir: str = DEFAULT_VIDEOS_DIR,
) -> Dict[str, Any]:
    """
    Downloads videos for target glosses from WLASL metadata.
    Prioritizes direct mp4 sources (e.g. aslbricks.org, signbank).
    """
    metadata_file = os.path.join(data_dir, "WLASL_v0.3.json")
    metadata = download_wlasl_metadata(metadata_file)
    if not metadata:
        return {
            "status": "error",
            "message": "Could not access or download WLASL metadata JSON.",
            "downloaded_videos": 0,
            "classes": [],
        }

    target_glosses_set = set(g.lower() for g in target_glosses)
    matching_glosses = [item for item in metadata if item.get("gloss", "").lower() in target_glosses_set]

    os.makedirs(videos_dir, exist_ok=True)
    download_records = []
    class_counts: Dict[str, int] = {}

    for item in matching_glosses:
        gloss = item.get("gloss", "").lower()
        instances = item.get("instances", [])
        class_counts[gloss] = 0

        # Sort instances: direct .mp4 URLs first (e.g. aslbricks.org)
        sorted_instances = sorted(
            instances,
            key=lambda x: (
                0 if "aslbricks.org" in x.get("url", "") else (
                    1 if x.get("url", "").endswith(".mp4") else 2
                )
            )
        )

        for inst in sorted_instances:
            if class_counts[gloss] >= max_instances_per_gloss:
                break

            video_id = inst.get("video_id") or inst.get("instance_id")
            video_url = inst.get("url", "")
            if not video_url or not video_url.startswith("http"):
                continue

            # Only direct video downloads are supported without external scrapers
            if not (video_url.endswith(".mp4") or "aslbricks" in video_url or "signbank" in video_url):
                continue

            filename = f"{gloss}_{video_id}.mp4"
            filepath = os.path.join(videos_dir, filename)

            logger.info(f"Downloading [{gloss}] ({video_id}) from {video_url}...")
            success = download_video_file(video_url, filepath)
            if success:
                class_counts[gloss] += 1
                download_records.append({
                    "gloss": gloss,
                    "video_id": str(video_id),
                    "filename": filename,
                    "filepath": filepath,
                    "url": video_url,
                    "split": inst.get("split", "train"),
                })
                logger.info(f"Saved: {filepath}")

    summary = {
        "status": "success",
        "dataset": "WLASL",
        "target_glosses": list(target_glosses),
        "total_downloaded_videos": len(download_records),
        "class_counts": class_counts,
        "videos_dir": videos_dir,
        "records": download_records,
    }

    info_path = os.path.join(data_dir, "word_dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Download complete: {len(download_records)} videos downloaded across {len(class_counts)} classes.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Download WLASL word-level sign dataset")
    parser.add_argument("--glosses", nargs="+", default=DEFAULT_TARGET_GLOSSES, help="Target sign glosses")
    parser.add_argument("--max_per_gloss", type=int, default=3, help="Max videos per sign")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Output directory")
    args = parser.parse_args()

    result = download_wlasl_subset(
        target_glosses=args.glosses,
        max_instances_per_gloss=args.max_per_gloss,
        data_dir=args.data_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

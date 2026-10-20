"""
Script to download the ASL American Sign Language Alphabet dataset from Kaggle using KaggleHub.
Identifier: debashishsau/aslamerican-sign-language-aplhabet-dataset
"""

import sys
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_HANDLE = "debashishsau/aslamerican-sign-language-aplhabet-dataset"
INFO_FILE_PATH = os.path.join("data", "dataset_info.json")


def download_asl_dataset(dataset_handle=DATASET_HANDLE):
    """
    Downloads the ASL dataset via kagglehub without hardcoding paths.
    Caches the path location in data/dataset_info.json for subsequent processing steps.
    """
    try:
        import kagglehub
    except ImportError:
        logger.error(
            "kagglehub is not installed. Please run: pip install kagglehub"
        )
        sys.exit(1)

    logger.info(f"Initiating dataset download via kagglehub for: {dataset_handle}")
    logger.info("If already cached locally by KaggleHub, it will resolve instantly.")

    try:
        # Download the dataset from Kaggle
        path = kagglehub.dataset_download(dataset_handle)
        print("Path to dataset files:", path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Downloaded path does not exist on disk: {path}")

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        # Save metadata about the download
        info = {
            "dataset_handle": dataset_handle,
            "dataset_path": path,
            "status": "downloaded"
        }
        with open(INFO_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Dataset path successfully recorded to {INFO_FILE_PATH}")
        return path

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("FAILED TO DOWNLOAD DATASET VIA KAGGLEHUB")
        logger.error("=" * 70)
        logger.error(f"Error details: {e}")
        logger.error("\nTroubleshooting checklist:")
        logger.error("1. Ensure you have an active internet connection.")
        logger.error("2. Verify that kagglehub has read/write permissions to its cache folder.")
        logger.error("3. If this dataset requires Kaggle credentials:")
        logger.error("   - Go to https://www.kaggle.com/settings -> 'Create New Token'")
        logger.error("   - Place the downloaded 'kaggle.json' in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\\.kaggle\\ (Windows)")
        logger.error("4. Check if the dataset identifier 'debashishsau/aslamerican-sign-language-aplhabet-dataset' is accessible.")
        logger.error("=" * 70 + "\n")
        raise e


if __name__ == "__main__":
    download_asl_dataset()

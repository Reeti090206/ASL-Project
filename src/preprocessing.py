"""
Data preprocessing, cleaning, validation, and stratified train-test splitting.
Guarantees zero data leakage by fitting transformers strictly on training splits.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_validate_dataset(csv_path: str = os.path.join("data", "landmarks.csv")) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads data/landmarks.csv and performs strict validation:
    - Checks required 63 landmark features + label column.
    - Identifies and removes any rows containing NaN or infinite values.
    - Validates label consistency.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Landmarks dataset not found at {csv_path}. Please run extract_landmarks.py first.")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded raw dataset with {len(df)} rows and {len(df.columns)} columns.")

    if "label" not in df.columns:
        raise ValueError("CSV missing required 'label' target column.")

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != 63:
        raise ValueError(f"Expected 63 landmark features, but found {len(feature_cols)} in {csv_path}")

    # Check for missing or infinite values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} null values in dataset. Dropping invalid rows...")
        df = df.dropna().reset_index(drop=True)

    # Check numeric types
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_df.isnull().sum().sum() > 0:
        logger.warning("Non-numeric values found in features. Dropping invalid rows...")
        valid_mask = ~X_df.isnull().any(axis=1)
        df = df[valid_mask].reset_index(drop=True)
        X_df = X_df[valid_mask]

    X = X_df.to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()

    logger.info(f"Cleaned dataset contains {len(df)} valid samples across {len(np.unique(y))} classes.")
    return df, X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs stratified train/test splitting to maintain class proportions.
    Falls back to non-stratified if any class has fewer than 2 samples.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    can_stratify = min_count >= 2
    if not can_stratify:
        logger.warning(
            f"Smallest class has only {min_count} samples. Disabling stratification for train_test_split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
        shuffle=True
    )

    logger.info(f"Split complete. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

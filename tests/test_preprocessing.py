import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import load_and_validate_dataset, split_dataset
from src.feature_extraction import FEATURE_COLUMNS


def test_load_and_validate_dataset():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        # Create valid synthetic dataset: 10 rows, 2 classes
        data = {"label": ["A"] * 5 + ["B"] * 5}
        for col in FEATURE_COLUMNS:
            data[col] = np.random.uniform(-1, 1, 10)

        df_orig = pd.DataFrame(data)
        # Add one row with NaN to verify automatic dropping
        nan_row = {"label": "A"}
        for col in FEATURE_COLUMNS:
            nan_row[col] = np.nan
        df_orig = pd.concat([df_orig, pd.DataFrame([nan_row])], ignore_index=True)
        df_orig.to_csv(csv_path, index=False)

        df, X, y = load_and_validate_dataset(csv_path)
        assert len(df) == 10  # NaN row was dropped
        assert X.shape == (10, 63)
        assert len(y) == 10
        assert set(np.unique(y)) == {"A", "B"}
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


def test_split_dataset_stratification():
    # 20 samples of A, 20 samples of B
    X = np.random.randn(40, 63).astype(np.float32)
    y = np.array(["A"] * 20 + ["B"] * 20)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 32
    assert len(X_test) == 8
    # Stratified: 4 A's and 4 B's in test set
    assert (y_test == "A").sum() == 4
    assert (y_test == "B").sum() == 4

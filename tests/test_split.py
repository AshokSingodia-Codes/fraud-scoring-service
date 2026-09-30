"""Unit tests for time-based dataset splitting and expanding-window CV."""

import numpy as np
import pandas as pd
import pytest
from src.fraud.data.split import (
    ExpandingWindowTimeSeriesSplit,
    save_split_info,
    time_split_data,
)


@pytest.fixture
def synthetic_time_df():
    n = 1000
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": np.arange(100, 100 + n * 10, 10),
            "isFraud": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
            "amount": np.random.uniform(10, 500, size=n),
        }
    )
    return df


def test_time_split_data(synthetic_time_df):
    train, val, test = time_split_data(synthetic_time_df, 0.70, 0.15, 0.15)

    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
    assert len(train) + len(val) + len(test) == len(synthetic_time_df)


def test_invalid_ratios(synthetic_time_df):
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        time_split_data(synthetic_time_df, 0.60, 0.15, 0.15)


def test_save_split_info(synthetic_time_df, tmp_path):
    train, val, test = time_split_data(synthetic_time_df)
    out_json = tmp_path / "split_info.json"

    info = save_split_info(train, val, test, output_path=out_json)

    assert out_json.exists()
    assert "train" in info
    assert "validation" in info
    assert "test" in info
    assert info["train"]["count"] == 700


def test_expanding_window_cv(synthetic_time_df):
    cv = ExpandingWindowTimeSeriesSplit(n_splits=4)
    splits = list(cv.split(synthetic_time_df))

    assert len(splits) == 4

    for fold, (tr_idx, val_idx) in enumerate(splits, 1):
        assert len(tr_idx) > 0
        assert len(val_idx) > 0
        assert tr_idx[-1] < val_idx[0]  # Strict time separation
        if fold > 1:
            prev_tr_idx, _ = splits[fold - 2]
            assert len(tr_idx) > len(prev_tr_idx)  # Expanding train set

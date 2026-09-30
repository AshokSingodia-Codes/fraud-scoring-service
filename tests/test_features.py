"""Unit tests for FeatureBuilder."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from src.fraud.features.builder import FeatureBuilder


@pytest.fixture
def sample_train_df():
    """Create tiny synthetic train DataFrame."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 0, 1, 0, 0],
            "TransactionDT": [86400, 86450, 87000, 90000, 95000],
            "TransactionAmt": [100.0, 55.5, 200.0, 150.0, 99.99],
            "ProductCD": ["W", "H", "W", "C", "W"],
            "card1": [1000, 1000, 2000, 1000, 3000],
            "addr1": [150, 150, 200, 150, 300],
            "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", "hotmail.com", None],
            "R_emaildomain": ["gmail.com", None, "gmail.com", None, None],
            "id_01": [-5.0, None, 0.0, None, -10.0],
            "id_30": ["Windows", "iOS", None, "Windows", "Android"],
        }
    )


@pytest.fixture
def sample_test_df():
    """Create tiny synthetic test DataFrame with unseen values."""
    return pd.DataFrame(
        {
            "TransactionID": [6],
            "TransactionDT": [100000],
            "TransactionAmt": [50.0],
            "ProductCD": ["R"],  # Unseen category
            "card1": [9999],   # Unseen card1
            "addr1": [999],    # Unseen addr1
            "P_emaildomain": ["unknown.com"],  # Unseen domain
            "R_emaildomain": [None],
            "id_01": [None],
            "id_30": ["MacOS"],
        }
    )


def test_feature_builder_fit_transform(sample_train_df, sample_test_df):
    fb = FeatureBuilder()
    fb.fit(sample_train_df)

    assert fb.is_fitted
    assert len(fb.final_feature_names) > 0

    train_ft = fb.transform(sample_train_df)
    test_ft = fb.transform(sample_test_df)

    assert list(train_ft.columns) == fb.final_feature_names
    assert list(test_ft.columns) == fb.final_feature_names
    assert "isFraud" not in train_ft.columns
    assert "TransactionID" not in train_ft.columns

    # Verify engineered features exist
    assert "day_index" in train_ft.columns
    assert "hour_of_day" in train_ft.columns
    assert "log_TransactionAmt" in train_ft.columns
    assert "card1_fq" in train_ft.columns
    assert "has_identity" in train_ft.columns


def test_feature_builder_unseen_values(sample_train_df, sample_test_df):
    fb = FeatureBuilder()
    fb.fit(sample_train_df)
    test_ft = fb.transform(sample_test_df)

    # Unseen card1 frequency should be 0
    assert test_ft["card1_fq"].iloc[0] == 0.0


def test_feature_builder_parity(sample_train_df):
    fb = FeatureBuilder()
    fb.fit(sample_train_df)

    batch_ft = fb.transform(sample_train_df)
    single_ft = fb.transform(sample_train_df.iloc[[0]])

    for col in fb.final_feature_names:
        v1 = batch_ft[col].iloc[0]
        v2 = single_ft[col].iloc[0]
        if pd.isna(v1):
            assert pd.isna(v2)
        else:
            assert v1 == v2


def test_feature_builder_save_load(sample_train_df):
    fb = FeatureBuilder()
    fb.fit(sample_train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        j_path = Path(tmpdir) / "fb.joblib"
        s_path = Path(tmpdir) / "spec.json"
        fb.save(j_path, s_path)

        loaded_fb = FeatureBuilder.load(j_path)
        assert loaded_fb.is_fitted
        assert loaded_fb.final_feature_names == fb.final_feature_names

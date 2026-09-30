"""Tests ensuring strict temporal non-leakage and partition independence."""

import pandas as pd
import pytest
from src.fraud.data.split import time_split_data


@pytest.fixture
def sample_dataset():
    n = 500
    return pd.DataFrame(
        {
            "TransactionID": [f"T_{i}" for i in range(n)],
            "TransactionDT": [86400 + i * 100 for i in range(n)],
            "isFraud": [1 if i % 20 == 0 else 0 for i in range(n)],
        }
    )


def test_no_temporal_leakage(sample_dataset):
    train, val, test = time_split_data(sample_dataset)

    max_train_dt = train["TransactionDT"].max()
    min_val_dt = val["TransactionDT"].min()
    max_val_dt = val["TransactionDT"].max()
    min_test_dt = test["TransactionDT"].min()

    assert max_train_dt < min_val_dt, f"Train max DT ({max_train_dt}) >= Val min DT ({min_val_dt})"
    assert max_val_dt < min_test_dt, f"Val max DT ({max_val_dt}) >= Test min DT ({min_test_dt})"


def test_no_id_overlap(sample_dataset):
    train, val, test = time_split_data(sample_dataset)

    train_ids = set(train["TransactionID"])
    val_ids = set(val["TransactionID"])
    test_ids = set(test["TransactionID"])

    assert len(train_ids & val_ids) == 0, "Overlap found between train and validation IDs"
    assert len(val_ids & test_ids) == 0, "Overlap found between validation and test IDs"
    assert len(train_ids & test_ids) == 0, "Overlap found between train and test IDs"


def test_fraud_rate_preservation(sample_dataset):
    train, val, test = time_split_data(sample_dataset)

    for name, df in [("Train", train), ("Validation", val), ("Test", test)]:
        fraud_rate = df["isFraud"].mean()
        assert fraud_rate > 0.0, f"{name} split has 0 fraud instances"
        print(f"{name} set fraud rate: {fraud_rate:.4%}")

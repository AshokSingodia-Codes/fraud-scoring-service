"""Tests for data ingestion and memory optimization."""

import numpy as np
import pandas as pd
from src.fraud.data.ingest import downcast_df, ingest_data


def test_downcast_df():
    df = pd.DataFrame(
        {
            "float_col": [1.0, 2.5, 3.14159],
            "int_col": [1, 2, 100],
            "str_col": ["a", "b", "c"],
        }
    )

    downcasted = downcast_df(df)

    assert downcasted["float_col"].dtype == np.float32
    assert downcasted["int_col"].dtype == np.uint8
    assert isinstance(downcasted["str_col"].dtype, pd.CategoricalDtype)


def test_ingest_data_with_synthetic_fixtures(tmp_path):
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    raw_dir.mkdir()
    interim_dir.mkdir()
    reports_dir.mkdir()

    # Synthetic transactions
    df_txn = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 1, 0, 0],
            "TransactionDT": [86400, 86405, 86500, 86600],
            "TransactionAmt": [10.0, 50.5, 100.0, 20.0],
            "card1": [1000, 2000, 1000, 3000],
            "C1": [1, 2, 1, 1],
        }
    )

    # Synthetic identity (exists for 2 of 4)
    df_id = pd.DataFrame(
        {
            "TransactionID": [2, 4],
            "id_01": [-5.0, 0.0],
            "DeviceType": ["mobile", "desktop"],
        }
    )

    df_txn.to_csv(raw_dir / "train_transaction.csv", index=False)
    df_id.to_csv(raw_dir / "train_identity.csv", index=False)

    config_path = tmp_path / "config.yaml"
    import yaml

    config_data = {
        "paths": {
            "raw_data_dir": str(raw_dir),
            "interim_data_dir": str(interim_dir),
            "reports_dir": str(reports_dir),
        }
    }
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    profile = ingest_data(config_path)

    assert profile["row_count"] == 4
    assert profile["column_count"] == 8
    assert profile["fraud_rate"] == 0.25
    assert profile["identity_share"] == 0.5
    assert (interim_dir / "train_merged.parquet").exists()
    assert (reports_dir / "data_profile.json").exists()

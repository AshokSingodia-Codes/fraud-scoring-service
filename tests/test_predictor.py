from pathlib import Path

import pytest
from src.fraud.serving.predictor import FraudPredictor


def test_predictor_initialization():
    predictor = FraudPredictor()
    assert predictor.feature_builder is not None
    assert predictor.booster is not None
    assert predictor.t_review > 0
    assert predictor.t_block > predictor.t_review


def test_load_every_artifact_and_score():
    """Verify that every model artifact loads cleanly and can score a transaction."""
    import json
    from pathlib import Path

    import joblib
    import lightgbm as lgb
    from src.fraud.utils.synthetic import generate_synthetic_transaction

    models_dir = Path("models")
    booster = lgb.Booster(model_file=str(models_dir / "model.txt"))
    assert booster is not None

    fb = joblib.load(models_dir / "feature_builder.joblib")
    assert fb.is_fitted

    cal = joblib.load(models_dir / "calibrator.joblib")
    assert cal.is_fitted

    with open(models_dir / "thresholds.json") as f:
        thresh = json.load(f)
        assert "t_review" in thresh and "t_block" in thresh

    with open(models_dir / "feature_spec.json") as f:
        spec = json.load(f)
        assert len(spec["final_feature_names"]) == 456

    predictor = FraudPredictor()
    tx = generate_synthetic_transaction("smoke_test")
    res = predictor.score_dict(tx, explain=True)
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["decision"] in ["approve", "review", "block"]
    assert len(res["reasons"]) == 5


def test_predictor_score_dict():
    predictor = FraudPredictor()
    sample_payload = {
        "transaction_id": "tx_test_1001",
        "TransactionDT": 86400,
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "card1": 1000,
        "addr1": 150,
        "P_emaildomain": "gmail.com",
    }

    res = predictor.score_dict(sample_payload, explain=True)

    assert res["transaction_id"] == "tx_test_1001"
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["decision"] in ["approve", "review", "block"]
    assert res["risk_band"] in ["Low", "Medium", "High"]
    assert len(res["reasons"]) > 0
    assert res["latency_ms"] > 0


def test_fast_path_feature_and_prediction_parity_on_200_samples():
    """Assert fast-path feature vector equals reference on 200 sample transactions and predictions match within 1e-6."""
    import numpy as np
    import pandas as pd
    from lightgbm.basic import _data_from_pandas
    from src.fraud.explain.reasons import explain_prediction
    from src.fraud.utils.synthetic import generate_synthetic_transaction

    predictor = FraudPredictor()
    txs = [generate_synthetic_transaction(f"tx_parity_{i}") for i in range(200)]

    # Batch reference feature matrix via DataFrame transform (vectorized across all 200 rows)
    raw_df = pd.DataFrame(txs)
    ref_df = predictor.feature_builder.transform(raw_df)
    mat_ref_all, _, _, _ = _data_from_pandas(
        ref_df, None, None, predictor.booster.pandas_categorical
    )

    for i, tx in enumerate(txs):
        mat_ref = mat_ref_all[[i]]
        # Fast-path feature vector via transform_dict_to_numpy
        mat_fast = predictor.feature_builder.transform_dict_to_numpy(tx)

        # 1. Feature vectors equality assertion (accounting for NaN locations across ALL 456 columns)
        np.testing.assert_array_equal(np.isnan(mat_ref), np.isnan(mat_fast))
        valid_mask = ~np.isnan(mat_ref)
        np.testing.assert_allclose(
            mat_ref[valid_mask],
            mat_fast[valid_mask],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Feature vector mismatch between old and optimized paths",
        )

        # 2. Prediction outputs match within 1e-6
        p_ref = float(predictor.booster.predict(mat_ref, num_threads=1)[0])
        p_fast = float(predictor.booster.predict(mat_fast, num_threads=1)[0])
        assert abs(p_ref - p_fast) < 1e-6, f"Prediction diff {abs(p_ref - p_fast)} exceeds 1e-6"

        # 3. Raw TreeSHAP contributions match within 1e-5 on all features
        c_ref = predictor.booster.predict(mat_ref, pred_contrib=True, num_threads=1)[0]
        c_fast = predictor.booster.predict(mat_fast, pred_contrib=True, num_threads=1)[0]
        np.testing.assert_allclose(c_ref, c_fast, rtol=1e-5, atol=1e-5)

        # 4. Explain formatted outputs match
        exp_ref = explain_prediction(predictor.booster, mat_ref, top_k=5, raw_payload=tx, num_threads=1)
        exp_fast = explain_prediction(
            predictor.booster, mat_fast, top_k=5, raw_payload=tx, feature_names=predictor.feature_names, num_threads=1
        )
        assert abs(exp_ref["margin_score"] - exp_fast["margin_score"]) < 1e-5
        assert [r["feature"] for r in exp_ref["reasons"]] == [r["feature"] for r in exp_fast["reasons"]]
        for r_ref, r_fast in zip(exp_ref["reasons"], exp_fast["reasons"]):
            assert abs(r_ref["contribution"] - r_fast["contribution"]) <= 1e-3
            assert r_ref["direction"] == r_fast["direction"]


def test_fast_path_edge_cases_parity():
    """Verify parity on tricky edge cases: missing card1/addr1, nodot email, None/empty strings,

    unseen categories, extra unknown keys, extreme TransactionDT, and missing optional blocks.
    """
    import numpy as np
    import pandas as pd
    from lightgbm.basic import _data_from_pandas
    from src.fraud.explain.reasons import explain_prediction

    predictor = FraudPredictor()

    edge_cases = [
        # 1. Missing card1/addr1
        {"transaction_id": "edge_missing_card_addr", "TransactionDT": 86400, "TransactionAmt": 50.0},
        # 2. Email domain with no dot
        {
            "transaction_id": "edge_nodot_email",
            "TransactionDT": 86400,
            "TransactionAmt": 50.0,
            "P_emaildomain": "nodotdomain",
            "R_emaildomain": "anothernodot",
        },
        # 3. None and empty-string values
        {
            "transaction_id": "edge_none_and_empty",
            "TransactionDT": 86400,
            "TransactionAmt": 50.0,
            "ProductCD": None,
            "P_emaildomain": "",
            "R_emaildomain": None,
            "DeviceInfo": "",
            "card4": None,
        },
        # 4. Unseen category values
        {
            "transaction_id": "edge_unseen_categories",
            "TransactionDT": 86400,
            "TransactionAmt": 50.0,
            "ProductCD": "XYZ_UNSEEN",
            "DeviceInfo": "AlienDevice 9000",
            "P_emaildomain": "mars.space.alien",
            "card6": "galactic_credit",
        },
        # 5. Extra unknown keys
        {
            "transaction_id": "edge_extra_keys",
            "TransactionDT": 86400,
            "TransactionAmt": 50.0,
            "extra_key_1": 12345,
            "unknown_metadata": "some_string",
        },
        # 6. TransactionDT of 0 and very large value
        {"transaction_id": "edge_dt_zero", "TransactionDT": 0, "TransactionAmt": 50.0},
        {"transaction_id": "edge_dt_large", "TransactionDT": 999999999, "TransactionAmt": 50.0},
        # 7. Payloads without any optional C/D/V/id blocks
        {"transaction_id": "edge_minimal_payload", "TransactionDT": 12345, "TransactionAmt": 99.99, "card1": 1000},
    ]

    for tx in edge_cases:
        ref_df = predictor.feature_builder.transform(pd.DataFrame([tx]))
        mat_ref, _, _, _ = _data_from_pandas(
            ref_df, None, None, predictor.booster.pandas_categorical
        )
        mat_fast = predictor.feature_builder.transform_dict_to_numpy(tx)

        # Assert exact NaN mask parity
        np.testing.assert_array_equal(np.isnan(mat_ref), np.isnan(mat_fast))
        valid_mask = ~np.isnan(mat_ref)
        np.testing.assert_allclose(mat_ref[valid_mask], mat_fast[valid_mask], rtol=1e-5, atol=1e-5)

        # Assert prediction parity within 1e-6
        p_ref = float(predictor.booster.predict(mat_ref, num_threads=1)[0])
        p_fast = float(predictor.booster.predict(mat_fast, num_threads=1)[0])
        assert abs(p_ref - p_fast) < 1e-6

        # Assert explain output parity
        exp_ref = explain_prediction(predictor.booster, mat_ref, top_k=5, raw_payload=tx, num_threads=1)
        exp_fast = explain_prediction(
            predictor.booster, mat_fast, top_k=5, raw_payload=tx, feature_names=predictor.feature_names, num_threads=1
        )
        assert abs(exp_ref["margin_score"] - exp_fast["margin_score"]) < 1e-6
        assert [r["feature"] for r in exp_ref["reasons"]] == [r["feature"] for r in exp_fast["reasons"]]
        for r_ref, r_fast in zip(exp_ref["reasons"], exp_fast["reasons"]):
            assert abs(r_ref["contribution"] - r_fast["contribution"]) < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("data/interim/train_merged.parquet").exists(),
    reason="Local parquet dataset data/interim/train_merged.parquet absent",
)
def test_fast_path_parity_on_500_real_rows():
    """Assert parity against 500 real test partition records if local data exists."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from lightgbm.basic import _data_from_pandas
    from src.fraud.explain.reasons import explain_prediction

    data_path = Path("data/interim/train_merged.parquet")
    predictor = FraudPredictor()

    table = pq.read_table(data_path, filters=[("TransactionDT", ">=", 152 * 86400)])
    test_df = table.slice(0, 500).to_pandas()
    records = test_df.to_dict(orient="records")

    for tx in records:
        clean_tx = {k: v for k, v in tx.items() if pd.notna(v)}

        ref_df = predictor.feature_builder.transform(pd.DataFrame([clean_tx]))
        mat_ref, _, _, _ = _data_from_pandas(
            ref_df, None, None, predictor.booster.pandas_categorical
        )
        mat_fast = predictor.feature_builder.transform_dict_to_numpy(clean_tx)

        np.testing.assert_array_equal(np.isnan(mat_ref), np.isnan(mat_fast))
        valid_mask = ~np.isnan(mat_ref)
        np.testing.assert_allclose(mat_ref[valid_mask], mat_fast[valid_mask], rtol=1e-5, atol=1e-5)

        p_ref = float(predictor.booster.predict(mat_ref, num_threads=1)[0])
        p_fast = float(predictor.booster.predict(mat_fast, num_threads=1)[0])
        assert abs(p_ref - p_fast) < 1e-6

        exp_ref = explain_prediction(predictor.booster, mat_ref, top_k=5, raw_payload=clean_tx, num_threads=1)
        exp_fast = explain_prediction(
            predictor.booster, mat_fast, top_k=5, raw_payload=clean_tx, feature_names=predictor.feature_names, num_threads=1
        )
        assert abs(exp_ref["margin_score"] - exp_fast["margin_score"]) < 1e-6
        assert [r["feature"] for r in exp_ref["reasons"]] == [r["feature"] for r in exp_fast["reasons"]]



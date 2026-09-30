"""Unit tests for baseline model training and evaluation functions."""

import numpy as np
import pandas as pd
from src.fraud.models.baseline import (
    train_dummy_baseline,
    train_lgbm_baseline,
    train_logistic_baseline,
)


def test_train_dummy_baseline():
    X_train = pd.DataFrame({"num": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 0, 1])  # 25% fraud
    X_val = pd.DataFrame({"num": [5, 6]})

    probs = train_dummy_baseline(X_train, y_train, X_val)

    assert len(probs) == 2
    assert np.allclose(probs, 0.25)


def test_train_logistic_baseline():
    X_train = pd.DataFrame({"num1": [1.0, 2.0, 10.0, 20.0], "num2": [0.1, 0.2, 1.0, 2.0]})
    y_train = pd.Series([0, 0, 1, 1])
    X_val = pd.DataFrame({"num1": [1.5, 15.0], "num2": [0.15, 1.5]})

    probs = train_logistic_baseline(X_train, y_train, X_val)

    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert probs[1] > probs[0]  # Higher feature values correspond to class 1


def test_train_lgbm_baseline():
    X_train = pd.DataFrame(
        {
            "num": [1.0, 2.0, 10.0, 20.0],
            "cat": pd.Categorical(["A", "A", "B", "B"]),
        }
    )
    y_train = pd.Series([0, 0, 1, 1])
    X_val = pd.DataFrame(
        {
            "num": [1.5, 15.0],
            "cat": pd.Categorical(["A", "B"]),
        }
    )

    probs = train_lgbm_baseline(X_train, y_train, X_val)

    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0

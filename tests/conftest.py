"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_config_dict():
    return {
        "project": {"name": "fraud-scoring-service", "seed": 42},
        "data": {"train_split": 0.70, "val_split": 0.15, "test_split": 0.15},
    }

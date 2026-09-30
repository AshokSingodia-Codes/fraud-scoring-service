"""Tests for config loading."""

from src.fraud.config import load_config


def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert config["project"]["name"] == "fraud-scoring-service"
    assert config["project"]["seed"] == 42
    assert config["data"]["train_split"] == 0.70

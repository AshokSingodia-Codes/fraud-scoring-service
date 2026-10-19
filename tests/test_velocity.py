"""Unit tests for the real-time velocity feature store (Redis & in-memory fallback)."""

from unittest.mock import MagicMock

from src.fraud.features.velocity import (
    InMemoryVelocityStore,
    RedisVelocityStore,
    get_velocity_store,
)


def test_in_memory_velocity_single_card():
    store = InMemoryVelocityStore()
    card = "12345"
    base_t = 100000.0

    # Initially empty
    feats = store.get_features(card, current_time=base_t)
    assert feats["card1_tx_cnt_10m"] == 0.0
    assert feats["card1_amt_sum_10m"] == 0.0
    assert feats["card1_tx_cnt_1h"] == 0.0
    assert feats["card1_amt_sum_1h"] == 0.0
    assert feats["card1_tx_cnt_24h"] == 0.0
    assert feats["card1_amt_sum_24h"] == 0.0

    # Record 1st transaction: $50 at base_t
    store.record_transaction(card, 50.0, current_time=base_t)

    # Check immediately after
    feats = store.get_features(card, current_time=base_t)
    assert feats["card1_tx_cnt_10m"] == 1.0
    assert feats["card1_amt_sum_10m"] == 50.0
    assert feats["card1_tx_cnt_1h"] == 1.0
    assert feats["card1_amt_sum_1h"] == 50.0
    assert feats["card1_tx_cnt_24h"] == 1.0
    assert feats["card1_amt_sum_24h"] == 50.0

    # Record 2nd transaction: $150 at base_t + 300s (5 minutes later)
    store.record_transaction(card, 150.0, current_time=base_t + 300.0)

    # Check at base_t + 300s
    feats = store.get_features(card, current_time=base_t + 300.0)
    assert feats["card1_tx_cnt_10m"] == 2.0
    assert feats["card1_amt_sum_10m"] == 200.0

    # Advance time to base_t + 700s (11.6 minutes later, so 1st transaction fell out of 10m window)
    feats = store.get_features(card, current_time=base_t + 700.0)
    assert feats["card1_tx_cnt_10m"] == 1.0
    assert feats["card1_amt_sum_10m"] == 150.0
    assert feats["card1_tx_cnt_1h"] == 2.0
    assert feats["card1_amt_sum_1h"] == 200.0

    # Advance time to base_t + 4000s (over 1 hour later, so both fell out of 1h window but inside 24h)
    feats = store.get_features(card, current_time=base_t + 4000.0)
    assert feats["card1_tx_cnt_10m"] == 0.0
    assert feats["card1_amt_sum_10m"] == 0.0
    assert feats["card1_tx_cnt_1h"] == 0.0
    assert feats["card1_amt_sum_1h"] == 0.0
    assert feats["card1_tx_cnt_24h"] == 2.0
    assert feats["card1_amt_sum_24h"] == 200.0


def test_in_memory_velocity_multiple_cards_isolation():
    store = InMemoryVelocityStore()
    card_a = "1001"
    card_b = "2002"
    base_t = 1000.0

    store.record_transaction(card_a, 100.0, current_time=base_t)
    store.record_transaction(card_b, 500.0, current_time=base_t)

    feats_a = store.get_features(card_a, current_time=base_t)
    feats_b = store.get_features(card_b, current_time=base_t)

    assert feats_a["card1_amt_sum_10m"] == 100.0
    assert feats_a["card1_tx_cnt_10m"] == 1.0
    assert feats_b["card1_amt_sum_10m"] == 500.0
    assert feats_b["card1_tx_cnt_10m"] == 1.0


def test_in_memory_velocity_none_or_missing_card():
    store = InMemoryVelocityStore()
    feats = store.get_features(None)
    assert feats["card1_tx_cnt_10m"] == 0.0
    assert feats["card1_amt_sum_10m"] == 0.0

    # Record None should be a no-op without error
    store.record_transaction(None, 100.0)
    assert store.get_features(None)["card1_tx_cnt_10m"] == 0.0


def test_redis_velocity_store_pipeline_mock():
    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline

    now = 100000.0
    # Simulate return of 2 entries in 24h window:
    # 1. $25 at now - 200s (inside 10m, 1h, 24h)
    # 2. $100 at now - 1800s (outside 10m, inside 1h, 24h)
    mock_pipeline.execute.return_value = [
        None,  # zremrangebyscore
        [("25.0:12345", now - 200.0), ("100.0:67890", now - 1800.0)],
    ]

    r_store = RedisVelocityStore(mock_redis)
    feats = r_store.get_features("card_999", current_time=now)

    assert feats["card1_tx_cnt_10m"] == 1.0
    assert feats["card1_amt_sum_10m"] == 25.0
    assert feats["card1_tx_cnt_1h"] == 2.0
    assert feats["card1_amt_sum_1h"] == 125.0
    assert feats["card1_tx_cnt_24h"] == 2.0
    assert feats["card1_amt_sum_24h"] == 125.0

    # Test record_transaction triggers zadd and expire
    mock_pipeline.execute.return_value = [1, True]
    r_store.record_transaction("card_999", 75.5, current_time=now)
    mock_pipeline.zadd.assert_called_once()
    mock_pipeline.expire.assert_called_once()


def test_get_velocity_store_singleton():
    s1 = get_velocity_store()
    s2 = get_velocity_store()
    assert s1 is s2

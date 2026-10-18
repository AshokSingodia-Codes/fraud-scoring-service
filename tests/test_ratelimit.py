"""Unit tests for sliding-window rate limiters (Redis & in-memory fallback)."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from src.fraud.serving.ratelimit import (
    InMemorySlidingWindow,
    RedisSlidingWindow,
    check_rate_limit,
)


def test_in_memory_sliding_window_allows_under_limit():
    limiter = InMemorySlidingWindow()
    client = "test_user_1"

    # Limit = 3 requests in 10 seconds
    assert limiter.is_allowed(client, max_requests=3, window_sec=10) is True
    assert limiter.is_allowed(client, max_requests=3, window_sec=10) is True
    assert limiter.is_allowed(client, max_requests=3, window_sec=10) is True

    # 4th request must be blocked
    assert limiter.is_allowed(client, max_requests=3, window_sec=10) is False


def test_in_memory_eviction_of_idle_keys():
    limiter = InMemorySlidingWindow(cleanup_interval_sec=1.0)

    # Populate active and idle clients
    limiter.records["active_user"] = [1050.0]  # active within cutoff (1040.0)
    limiter.records["idle_user_1"] = [1010.0]  # idle (older than 1040.0)
    limiter.records["idle_user_empty"] = []    # empty records

    cutoff = 1040.0
    evicted_count = limiter.evict_idle_keys(now=1100.0, cutoff=cutoff)

    assert evicted_count == 2
    assert "idle_user_1" not in limiter.records
    assert "idle_user_empty" not in limiter.records
    assert "active_user" in limiter.records


def test_in_memory_sliding_window_resets_after_window():
    limiter = InMemorySlidingWindow()
    client = "test_user_expire"

    # Inject timestamps older than the 5-second window
    old_time = time.time() - 10.0
    limiter.records[client] = [old_time, old_time + 1.0]

    # New request should succeed because previous timestamps expired
    assert limiter.is_allowed(client, max_requests=2, window_sec=5) is True


def test_check_rate_limit_raises_http_429():
    client = "abusive_client"
    # Consume 2 requests
    check_rate_limit(client, max_requests=2, window_sec=60)
    check_rate_limit(client, max_requests=2, window_sec=60)

    # 3rd request should trigger HTTPException 429
    with pytest.raises(HTTPException) as exc_info:
        check_rate_limit(client, max_requests=2, window_sec=60)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail


def test_redis_sliding_window_pipeline():
    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline

    # Scenario 1: count is 2 (below limit of 5)
    mock_pipeline.execute.side_effect = [
        [None, 2],  # zremrangebyscore, zcard
        [1, True],  # zadd, expire
    ]

    limiter = RedisSlidingWindow(mock_redis)
    allowed = limiter.is_allowed("client_redis_1", max_requests=5, window_sec=60)

    assert allowed is True
    mock_pipeline.zremrangebyscore.assert_called_once()
    mock_pipeline.zcard.assert_called_once()
    mock_pipeline.zadd.assert_called_once()
    mock_pipeline.expire.assert_called_once()


def test_redis_sliding_window_blocks_on_limit():
    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline

    # Scenario 2: count is 5 (at limit of 5)
    mock_pipeline.execute.return_value = [None, 5]

    limiter = RedisSlidingWindow(mock_redis)
    allowed = limiter.is_allowed("client_redis_blocked", max_requests=5, window_sec=60)

    assert allowed is False

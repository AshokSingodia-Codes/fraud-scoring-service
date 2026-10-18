"""Rate limiting middleware and utilities (Stage 10)."""

import logging
import os
import time

import redis
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "5000"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))


class InMemorySlidingWindow:
    """Sliding-window in-memory limiter with automatic eviction of idle client keys."""

    def __init__(self, cleanup_interval_sec: float = 60.0):
        self.records: dict[str, list[float]] = {}
        self.last_cleanup: float = time.time()
        self.cleanup_interval: float = cleanup_interval_sec

    def evict_idle_keys(self, now: float, cutoff: float) -> int:
        """Evict client records that have no timestamps within the active window."""
        idle_keys = [
            cid for cid, timestamps in self.records.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for cid in idle_keys:
            del self.records[cid]
        self.last_cleanup = now
        return len(idle_keys)

    def is_allowed(self, client_id: str, max_requests: int, window_sec: int) -> bool:
        now = time.time()
        cutoff = now - window_sec

        # Periodic eviction of idle keys across all clients
        if now - self.last_cleanup > self.cleanup_interval:
            self.evict_idle_keys(now, cutoff)

        timestamps = [t for t in self.records.get(client_id, []) if t > cutoff]
        if len(timestamps) >= max_requests:
            self.records[client_id] = timestamps
            return False

        timestamps.append(now)
        self.records[client_id] = timestamps
        return True

    def reset(self) -> None:
        self.records.clear()
        self.last_cleanup = time.time()


class RedisSlidingWindow:
    """Atomic sliding-window rate limiter powered by Redis sorted sets (ZSET)."""

    def __init__(self, redis_client: redis.Redis):
        self.client = redis_client

    def is_allowed(self, client_id: str, max_requests: int, window_sec: int) -> bool:
        key = f"ratelimit:{client_id}"
        now = time.time()
        cutoff = now - window_sec
        member = f"{now}:{time.time_ns()}"

        pipe = self.client.pipeline()
        pipe.zremrangebyscore(key, "-inf", cutoff)
        pipe.zcard(key)
        results = pipe.execute()
        current_count = results[1]

        if current_count >= max_requests:
            return False

        pipe2 = self.client.pipeline()
        pipe2.zadd(key, {member: now})
        pipe2.expire(key, window_sec + 5)
        pipe2.execute()
        return True


_in_memory_limiter = InMemorySlidingWindow()
_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis | None:
    """Retrieve or create Redis client instance if REDIS_URL is configured."""
    global _redis_client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None

    if _redis_client is None:
        try:
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            _redis_client = client
        except Exception as e:
            logger.warning(
                "Failed to connect to Redis at %s (%s). Falling back to in-memory rate limiter.",
                redis_url,
                e,
            )
            _redis_client = None
    return _redis_client


def check_rate_limit(
    client_id: str,
    max_requests: int | None = None,
    window_sec: int | None = None,
) -> None:
    """Enforce sliding-window rate limit using Redis when available, with in-memory fallback."""
    max_req = max_requests if max_requests is not None else int(os.getenv("RATE_LIMIT_REQUESTS", "5000"))
    win_sec = window_sec if window_sec is not None else int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

    allowed = True
    r_client = get_redis_client()
    if r_client is not None:
        try:
            redis_limiter = RedisSlidingWindow(r_client)
            allowed = redis_limiter.is_allowed(client_id, max_req, win_sec)
        except Exception as e:
            logger.warning("Redis rate limit check error: %s. Using in-memory fallback.", e)
            allowed = _in_memory_limiter.is_allowed(client_id, max_req, win_sec)
    else:
        allowed = _in_memory_limiter.is_allowed(client_id, max_req, win_sec)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again in a minute.",
        )

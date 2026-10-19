"""Real-time online rolling velocity feature store (Stage 11).

Provides microsecond sliding-window aggregation features over transaction streams
using Redis sorted sets (ZSET) with automatic in-memory fallback.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

WINDOWS_SEC = {
    "10m": 600,
    "1h": 3600,
    "24h": 86400,
}


class VelocityStoreBase:
    """Abstract interface for velocity feature stores."""

    def get_features(self, entity_id: str | None, current_time: float | None = None) -> dict[str, float]:
        raise NotImplementedError

    def record_transaction(self, entity_id: str | None, amount: float, current_time: float | None = None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class InMemoryVelocityStore(VelocityStoreBase):
    """Thread-safe sliding-window in-memory velocity feature store."""

    def __init__(self, max_history_sec: float = 86400.0):
        self._records: dict[str, list[tuple[float, float]]] = {}  # entity_id -> list of (timestamp, amount)
        self._lock = threading.Lock()
        self.max_history_sec = max_history_sec

    def get_features(self, entity_id: str | None, current_time: float | None = None) -> dict[str, float]:
        if not entity_id:
            return {
                "card1_tx_cnt_10m": 0.0,
                "card1_amt_sum_10m": 0.0,
                "card1_tx_cnt_1h": 0.0,
                "card1_amt_sum_1h": 0.0,
                "card1_tx_cnt_24h": 0.0,
                "card1_amt_sum_24h": 0.0,
            }

        now = current_time if current_time is not None else time.time()
        t_10m = now - WINDOWS_SEC["10m"]
        t_1h = now - WINDOWS_SEC["1h"]
        t_24h = now - WINDOWS_SEC["24h"]

        cnt_10m, sum_10m = 0.0, 0.0
        cnt_1h, sum_1h = 0.0, 0.0
        cnt_24h, sum_24h = 0.0, 0.0

        with self._lock:
            items = self._records.get(str(entity_id), [])
            for ts, amt in items:
                if ts >= t_24h:
                    cnt_24h += 1.0
                    sum_24h += amt
                    if ts >= t_1h:
                        cnt_1h += 1.0
                        sum_1h += amt
                        if ts >= t_10m:
                            cnt_10m += 1.0
                            sum_10m += amt

        return {
            "card1_tx_cnt_10m": cnt_10m,
            "card1_amt_sum_10m": round(sum_10m, 2),
            "card1_tx_cnt_1h": cnt_1h,
            "card1_amt_sum_1h": round(sum_1h, 2),
            "card1_tx_cnt_24h": cnt_24h,
            "card1_amt_sum_24h": round(sum_24h, 2),
        }

    def record_transaction(self, entity_id: str | None, amount: float, current_time: float | None = None) -> None:
        if not entity_id:
            return
        now = current_time if current_time is not None else time.time()
        cutoff = now - self.max_history_sec

        with self._lock:
            eid = str(entity_id)
            if eid not in self._records:
                self._records[eid] = []
            history = self._records[eid]
            # Prune events older than 24h
            valid_history = [item for item in history if item[0] >= cutoff]
            valid_history.append((now, float(amount)))
            self._records[eid] = valid_history

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


class RedisVelocityStore(VelocityStoreBase):
    """High-throughput sliding-window velocity store using Redis sorted sets (ZSET)."""

    def __init__(self, redis_client: Any, prefix: str = "vel:card1"):
        self.client = redis_client
        self.prefix = prefix

    def _key(self, entity_id: str) -> str:
        return f"{self.prefix}:{entity_id}"

    def get_features(self, entity_id: str | None, current_time: float | None = None) -> dict[str, float]:
        if not entity_id:
            return {
                "card1_tx_cnt_10m": 0.0,
                "card1_amt_sum_10m": 0.0,
                "card1_tx_cnt_1h": 0.0,
                "card1_amt_sum_1h": 0.0,
                "card1_tx_cnt_24h": 0.0,
                "card1_amt_sum_24h": 0.0,
            }

        now = current_time if current_time is not None else time.time()
        t_10m = now - WINDOWS_SEC["10m"]
        t_1h = now - WINDOWS_SEC["1h"]
        t_24h = now - WINDOWS_SEC["24h"]
        key = self._key(str(entity_id))

        # Query all events in 24h window
        pipe = self.client.pipeline()
        pipe.zremrangebyscore(key, "-inf", f"({t_24h}")
        pipe.zrangebyscore(key, t_24h, "+inf", withscores=True)
        results = pipe.execute()
        entries = results[1]  # list of (member, score)

        cnt_10m, sum_10m = 0.0, 0.0
        cnt_1h, sum_1h = 0.0, 0.0
        cnt_24h, sum_24h = 0.0, 0.0

        for member, ts in entries:
            # member format: "{amount}:{uuid_or_nanos}"
            try:
                amt_str = member.split(":")[0]
                amt = float(amt_str)
            except Exception:
                amt = 0.0

            cnt_24h += 1.0
            sum_24h += amt
            if ts >= t_1h:
                cnt_1h += 1.0
                sum_1h += amt
                if ts >= t_10m:
                    cnt_10m += 1.0
                    sum_10m += amt

        return {
            "card1_tx_cnt_10m": cnt_10m,
            "card1_amt_sum_10m": round(sum_10m, 2),
            "card1_tx_cnt_1h": cnt_1h,
            "card1_amt_sum_1h": round(sum_1h, 2),
            "card1_tx_cnt_24h": cnt_24h,
            "card1_amt_sum_24h": round(sum_24h, 2),
        }

    def record_transaction(self, entity_id: str | None, amount: float, current_time: float | None = None) -> None:
        if not entity_id:
            return
        now = current_time if current_time is not None else time.time()
        key = self._key(str(entity_id))
        member = f"{float(amount)}:{time.time_ns()}"

        pipe = self.client.pipeline()
        pipe.zadd(key, {member: now})
        pipe.expire(key, WINDOWS_SEC["24h"] + 60)
        pipe.execute()

    def reset(self) -> None:
        # Warning: For test environments only
        keys = self.client.keys(f"{self.prefix}:*")
        if keys:
            self.client.delete(*keys)


_global_velocity_store: VelocityStoreBase | None = None


def get_velocity_store() -> VelocityStoreBase:
    """Retrieve or initialize singleton velocity feature store (Redis if configured, else in-memory)."""
    global _global_velocity_store
    if _global_velocity_store is not None:
        return _global_velocity_store

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("VelocityFeatureStore connected to Redis at %s", redis_url)
            _global_velocity_store = RedisVelocityStore(client)
            return _global_velocity_store
        except Exception as e:
            logger.warning("Failed to connect to Redis for velocity store: %s. Using InMemory fallback.", e)

    _global_velocity_store = InMemoryVelocityStore()
    return _global_velocity_store

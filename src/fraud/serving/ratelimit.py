import os
import time
from collections import defaultdict

from fastapi import HTTPException, status

# In-memory token bucket rate limiter: default 5000 requests / 60 seconds
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "5000"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

request_records = defaultdict(list)


def check_rate_limit(client_id: str) -> None:
    """Enforce rate limits per API key / IP."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SEC

    # Clean old requests
    timestamps = [t for t in request_records[client_id] if t > cutoff]
    request_records[client_id] = timestamps

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again in a minute.",
        )

    request_records[client_id].append(now)

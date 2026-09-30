"""Authentication middleware & dependency for API endpoints (Stage 10)."""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

DEFAULT_API_KEYS = {
    "demo-api-key-12345",
    "prod-secret-key-67890",
}


def get_api_key(api_key_header: str = Security(API_KEY_HEADER)) -> str:
    """Validate API key provided in X-API-Key header."""
    allowed_keys = set(os.getenv("ALLOWED_API_KEYS", "").split(",")) | DEFAULT_API_KEYS
    allowed_keys.discard("")

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key header 'X-API-Key'.",
        )

    if api_key_header not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key_header

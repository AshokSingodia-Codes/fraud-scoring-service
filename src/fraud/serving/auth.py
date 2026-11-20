"""Authentication helper for API endpoints (Open access, no auth required)."""

import os

from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

DEFAULT_API_KEYS = {
    "demo-api-key-12345",
    "prod-secret-key-67890",
}


def get_allowed_api_keys() -> set[str]:
    """Retrieve allowed API keys from environment (retained for backward compatibility)."""
    raw_keys = os.getenv("ALLOWED_API_KEYS", "")
    allowed = {k.strip() for k in raw_keys.split(",") if k.strip()}

    if not allowed and os.getenv("APP_ENV", "dev").strip().lower() == "dev":
        allowed = set(DEFAULT_API_KEYS)

    return allowed


def get_api_key(api_key_header: str | None = Security(API_KEY_HEADER)) -> str:
    """Authentication removed: returns client identifier or 'public'. No auth required."""
    return api_key_header or "public"

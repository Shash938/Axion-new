"""
security/rate_limiter.py — Sliding Window Rate Limiting Module
===============================================================
In-memory sliding-window rate limiter protecting API endpoints against abuse.
Tracks timestamps of requests per client key / IP within a 60-second window.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import Depends, HTTPException, Request, Response, status

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """
    Sliding window rate limiter using timestamps per client key.
    """

    def __init__(self) -> None:
        # Client key -> List of request timestamps (float)
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._window_seconds: float = 60.0

    def clear(self) -> None:
        """Resets all rate limit tracking history (useful for testing)."""
        self._requests.clear()

    def is_rate_limited(self, client_key: str, max_requests: int) -> Tuple[bool, int, int, int]:
        """
        Checks if client_key has exceeded max_requests within window.
        Returns tuple: (is_limited, limit, remaining, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self._window_seconds

        # Prune old timestamps
        timestamps = [ts for ts in self._requests[client_key] if ts > window_start]
        self._requests[client_key] = timestamps

        current_count = len(timestamps)

        if current_count >= max_requests:
            oldest_ts = timestamps[0]
            retry_after = max(1, int(oldest_ts + self._window_seconds - now))
            remaining = 0
            return True, max_requests, remaining, retry_after

        # Record this request
        timestamps.append(now)
        self._requests[client_key] = timestamps
        remaining = max_requests - len(timestamps)
        return False, max_requests, remaining, 0


# Shared singleton rate limiter
_rate_limiter = InMemoryRateLimiter()


def reset_rate_limiter() -> None:
    """Helper function to reset rate limiter storage between unit tests."""
    _rate_limiter.clear()


def rate_limit_check(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> None:
    """
    FastAPI dependency enforcing rate limits per client IP / API key.
    Attaches rate limit headers to the response.
    """
    max_limit = settings.RATE_LIMIT_PER_MINUTE

    # Identify client by X-Forwarded-For header or direct client IP
    client_ip = request.client.host if request.client else "unknown-client"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    api_key = request.headers.get("X-API-Key")
    client_key = f"key:{api_key}" if api_key else f"ip:{client_ip}"

    is_limited, limit, remaining, retry_after = _rate_limiter.is_rate_limited(
        client_key=client_key,
        max_requests=max_limit,
    )

    # Set response headers
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    reset_timestamp = int(time.time()) + 60
    response.headers["X-RateLimit-Reset"] = str(reset_timestamp)

    if is_limited:
        logger.warning("Rate limit exceeded for client %s", client_key)
        response.headers["Retry-After"] = str(retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "type": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429",
                "title": "Too Many Requests",
                "status": 429,
                "detail": f"Rate limit exceeded. Maximum allowed: {limit} requests per minute.",
                "retry_after": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
            },
        )

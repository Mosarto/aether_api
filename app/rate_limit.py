"""
Per-user in-memory sliding window rate limiter.

Limitations:
- State is lost on process restart (in-memory only).
- Single-worker only — not safe for multi-process deployments.
  For multi-worker or distributed setups, replace with Redis-backed storage.

Usage in route handlers (after auth dependency resolves uid):
    await check_rate_limit(user["uid"])
"""

import asyncio
import time

from fastapi import HTTPException

from app.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


class RateLimitExceeded(HTTPException):
    """429 with top-level JSON body (not wrapped in 'detail')."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "retryAfter": retry_after},
            headers={"Retry-After": str(retry_after)},
        )

# uid -> list of request timestamps (epoch seconds)
_user_requests: dict[str, list[float]] = {}
_lock = asyncio.Lock()

# Sweep all users every N calls to prevent unbounded memory growth
_CLEANUP_INTERVAL = 100
_call_counter = 0


async def check_rate_limit(uid: str) -> None:
    """
    Sliding window rate limiter. Call after auth, before LLM operations.

    Args:
        uid: Authenticated user ID extracted from auth dependency.

    Raises:
        HTTPException 429: When user exceeds RATE_LIMIT_REQUESTS within
                           RATE_LIMIT_WINDOW_SECONDS. Includes Retry-After header.
    """
    global _call_counter

    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    async with _lock:
        _call_counter += 1

        # Per-user: prune timestamps outside the sliding window
        timestamps = _user_requests.get(uid, [])
        timestamps = [t for t in timestamps if t > window_start]

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            # Oldest timestamp in window determines when the slot frees up
            retry_after = int(timestamps[0] - window_start) + 1
            raise RateLimitExceeded(retry_after)

        timestamps.append(now)
        _user_requests[uid] = timestamps

        # Global sweep: remove users with no recent activity
        if _call_counter % _CLEANUP_INTERVAL == 0:
            cutoff = now - RATE_LIMIT_WINDOW_SECONDS
            empty = [u for u, ts in _user_requests.items() if not ts or ts[-1] <= cutoff]
            for u in empty:
                del _user_requests[u]

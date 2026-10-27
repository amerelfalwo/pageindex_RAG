"""Rate Limiter Module for Atlas API.

Provides a thread-safe, in-memory sliding-window rate limiter.
Dual-keyed on client_id AND client IP to prevent:
1. Limit bypass via rotating client UUIDs (guarded by IP bucket).
2. Shared bucket starvation behind proxies (guarded by client_id bucket).

Behind Hugging Face Spaces / reverse proxies, request.client.host points to the
internal proxy infrastructure. The originating client IP is extracted from the
first hop of the X-Forwarded-For header.
"""

import os
import threading
import time
from typing import Dict, List, Tuple, Optional
from fastapi import Request, HTTPException, status
from utils.logger import app_logger


def get_client_ip(request: Request) -> str:
    """
    Extracts the originating client IP address.
    Behind Hugging Face / reverse proxies, request.client.host is the internal proxy IP.
    The first hop in X-Forwarded-For contains the true client IP.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # First hop is the original client IP before traversing upstream proxies
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter with sub-second precision."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: Dict[str, List[float]] = {}

    def check_and_record(
        self,
        key: str,
        limit: int,
    ) -> Tuple[bool, int]:
        """
        Checks if key is within limit. If allowed, records the hit.
        Returns (is_allowed, retry_after_seconds).
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._hits.get(key, [])
            # Evict timestamps outside sliding window
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= limit:
                # Oldest timestamp in window determines remaining cooldown
                oldest_hit = timestamps[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest_hit) + 0.99))
                self._hits[key] = timestamps
                return False, retry_after

            timestamps.append(now)
            self._hits[key] = timestamps
            return True, 0

    def reset(self) -> None:
        """Clear all recorded hits (useful for test isolation)."""
        with self._lock:
            self._hits.clear()


# Global rate limiter instance
_global_limiter = SlidingWindowRateLimiter(window_seconds=60.0)


def get_rate_limiter() -> SlidingWindowRateLimiter:
    return _global_limiter


class DualKeyRateLimitChecker:
    """
    FastAPI dependency callable that checks rate limits against both
    client_id and originating IP address.
    """

    def __init__(self, action_name: str, default_limit: int, env_var_name: str):
        self.action_name = action_name
        self.default_limit = default_limit
        self.env_var_name = env_var_name

    def _get_limit(self) -> int:
        raw = os.getenv(self.env_var_name)
        if raw:
            try:
                val = int(raw.strip())
                if val > 0:
                    return val
            except ValueError:
                pass
        return self.default_limit

    async def __call__(self, request: Request) -> None:
        client_id = request.headers.get("x-client-id", "").strip().lower()
        ip_addr = get_client_ip(request)
        limit = self._get_limit()

        limiter = get_rate_limiter()

        # Check client_id bucket if present
        if client_id:
            client_key = f"client:{self.action_name}:{client_id}"
            allowed, retry_after = limiter.check_and_record(client_key, limit)
            if not allowed:
                app_logger.warning(
                    f"[RateLimit] Client rate limit exceeded for {client_id} on {self.action_name} "
                    f"({limit}/min). IP: {ip_addr}"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for {self.action_name}. Maximum {limit} requests per minute.",
                    headers={"Retry-After": str(retry_after)},
                )

        # Check IP bucket
        ip_key = f"ip:{self.action_name}:{ip_addr}"
        allowed_ip, retry_after_ip = limiter.check_and_record(ip_key, limit)
        if not allowed_ip:
            app_logger.warning(
                f"[RateLimit] IP rate limit exceeded for {ip_addr} on {self.action_name} "
                f"({limit}/min). Client ID: {client_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {self.action_name}. Maximum {limit} requests per minute.",
                headers={"Retry-After": str(retry_after_ip)},
            )

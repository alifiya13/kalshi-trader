"""
Token Bucket Rate Limiter

Prevents hitting Kalshi's rate limits (default Basic tier: 20 read/s, 10 write/s).
We target 80% of the limit as a safety margin.

Usage:
    limiter = RateLimiter(max_per_second=16)  # 80% of 20
    limiter.wait()  # blocks until a token is available
    # ...make request...
"""

import time
import threading


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, max_per_second: float):
        self.max_tokens = max_per_second
        self.tokens = max_per_second
        self.rate = max_per_second  # refill rate per second
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def wait(self):
        """Block until a token is available, then consume it."""
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            # Sleep a short interval before retrying
            time.sleep(0.02)

    def try_acquire(self) -> bool:
        """Non-blocking: returns True if a token was consumed, False otherwise."""
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

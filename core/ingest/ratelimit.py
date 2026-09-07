"""Thread-safe rate limiting for vendor API calls.

A worker pool without a shared limiter is not a rate limiter: eight threads each
sleeping 120ms between their own calls will happily emit eight times the intended
rate. The bucket here is shared by every worker and hands out permits against one
wall clock, so the ceiling holds regardless of pool size.

The limiter is also *adaptive*. A vendor's published limit and its enforced limit
differ (bursts, per-endpoint sub-limits, noisy-neighbour throttling), so a run
that never sees HTTP 429 is leaving throughput on the table while a run that sees
many is wasting calls on retries. :meth:`TokenBucket.penalize` halves the rate on
a throttle response and it recovers geometrically, which keeps a long backfill
near the true ceiling without a human tuning it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Floor and ceiling on the adaptive rate, as a fraction of the configured rate.
_MIN_RATE_FRACTION = 0.05
_RECOVERY_FACTOR = 1.05
_PENALTY_FACTOR = 0.5


@dataclass
class RateLimitStats:
    """Counters describing how a bucket behaved over a run."""

    permits: int = 0
    penalties: int = 0
    total_wait_seconds: float = 0.0
    current_rate_per_minute: float = 0.0


class TokenBucket:
    """
    Shared permit source that paces all workers to a calls-per-minute ceiling.

    Args:
        rate_per_minute: Target sustained call rate.
        adaptive: When True, throttle responses reduce the rate and success
            gradually restores it.

    Example:
        >>> bucket = TokenBucket(480)
        >>> bucket.acquire()  # blocks just long enough to stay under the ceiling
    """

    def __init__(self, rate_per_minute: float, adaptive: bool = True) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be positive, got {rate_per_minute}")
        self._configured_rate = float(rate_per_minute)
        self._rate = float(rate_per_minute)
        self._adaptive = adaptive
        self._lock = threading.Lock()
        self._next_slot = time.monotonic()
        self._stats = RateLimitStats(current_rate_per_minute=self._rate)

    def acquire(self) -> float:
        """
        Block until the caller may issue one request.

        Returns:
            Seconds spent waiting, for accounting.
        """
        with self._lock:
            now = time.monotonic()
            slot = max(self._next_slot, now)
            interval = 60.0 / self._rate
            self._next_slot = slot + interval
            self._stats.permits += 1
            wait = slot - now
            self._stats.total_wait_seconds += max(wait, 0.0)

        if wait > 0:
            time.sleep(wait)
        return max(wait, 0.0)

    def penalize(self) -> None:
        """Halve the rate after a throttle response, down to a floor."""
        if not self._adaptive:
            return
        with self._lock:
            floor = self._configured_rate * _MIN_RATE_FRACTION
            new_rate = max(floor, self._rate * _PENALTY_FACTOR)
            if new_rate < self._rate:
                logger.warning("rate limit penalty: %.0f -> %.0f calls/min", self._rate, new_rate)
            self._rate = new_rate
            self._stats.penalties += 1
            self._stats.current_rate_per_minute = self._rate

    def recover(self) -> None:
        """Nudge the rate back toward the configured ceiling after a success."""
        if not self._adaptive:
            return
        with self._lock:
            if self._rate >= self._configured_rate:
                return
            self._rate = min(self._configured_rate, self._rate * _RECOVERY_FACTOR)
            self._stats.current_rate_per_minute = self._rate

    @property
    def stats(self) -> RateLimitStats:
        """Snapshot of the bucket's counters."""
        with self._lock:
            return RateLimitStats(
                permits=self._stats.permits,
                penalties=self._stats.penalties,
                total_wait_seconds=self._stats.total_wait_seconds,
                current_rate_per_minute=self._rate,
            )

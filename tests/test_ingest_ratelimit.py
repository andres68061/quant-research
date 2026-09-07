"""
Unit tests for core.ingest.ratelimit.

The bucket is the only thing standing between a twelve-thread pool and a vendor
ban, so the properties tested here are: the shared ceiling actually holds when
many threads acquire at once, a throttle response lowers the rate but never below
the floor, recovery walks back up without ever exceeding the configured rate, and
a non-adaptive bucket is inert.

Timing assertions use the theoretical minimum as a lower bound (the algorithm
guarantees it) and a generous multiple of the target as the upper bound, so they
do not flake on a loaded CI box.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.ingest.ratelimit import RateLimitStats, TokenBucket


def _drain_concurrently(bucket: TokenBucket, n_calls: int, workers: int) -> float:
    """Acquire ``n_calls`` permits from ``workers`` threads; return elapsed seconds."""

    def one(_index: int) -> float:
        return bucket.acquire()

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(n_calls)))
    return time.monotonic() - started


def test_acquire_holds_target_rate_under_thread_pool() -> None:
    rate_per_minute = 3000.0
    n_calls = 24
    workers = 8
    interval = 60.0 / rate_per_minute

    bucket = TokenBucket(rate_per_minute)
    elapsed = _drain_concurrently(bucket, n_calls, workers)

    # The first permit is free; the remaining ones are spaced by `interval`.
    theoretical_minimum = (n_calls - 1) * interval
    assert elapsed >= theoretical_minimum * 0.9
    observed_rate = n_calls / elapsed * 60.0
    assert observed_rate <= rate_per_minute * 1.5
    assert bucket.stats.permits == n_calls


def test_acquire_returns_nonnegative_wait_and_counts_permits() -> None:
    bucket = TokenBucket(60_000)
    waits = [bucket.acquire() for _ in range(3)]

    assert all(isinstance(wait, float) for wait in waits)
    assert all(wait >= 0.0 for wait in waits)
    stats = bucket.stats
    assert isinstance(stats, RateLimitStats)
    assert stats.permits == 3
    assert stats.total_wait_seconds >= 0.0


def test_penalize_lowers_rate_and_respects_floor() -> None:
    bucket = TokenBucket(1000)

    bucket.penalize()
    assert bucket.stats.current_rate_per_minute == pytest.approx(500.0)

    for _ in range(50):
        bucket.penalize()

    floor = 1000 * 0.05
    assert bucket.stats.current_rate_per_minute == pytest.approx(floor)
    assert bucket.stats.current_rate_per_minute >= floor
    assert bucket.stats.penalties == 51


def test_recover_restores_toward_but_never_above_configured_rate() -> None:
    configured = 1000.0
    bucket = TokenBucket(configured)
    bucket.penalize()
    penalized_rate = bucket.stats.current_rate_per_minute

    bucket.recover()
    once_recovered = bucket.stats.current_rate_per_minute
    assert penalized_rate < once_recovered < configured

    for _ in range(200):
        bucket.recover()
        assert bucket.stats.current_rate_per_minute <= configured

    assert bucket.stats.current_rate_per_minute == pytest.approx(configured)


def test_recover_is_a_noop_at_the_configured_rate() -> None:
    bucket = TokenBucket(750)

    bucket.recover()

    assert bucket.stats.current_rate_per_minute == pytest.approx(750.0)


def test_penalize_is_ignored_by_a_non_adaptive_bucket() -> None:
    bucket = TokenBucket(1000, adaptive=False)

    bucket.penalize()
    bucket.penalize()
    bucket.recover()

    assert bucket.stats.current_rate_per_minute == pytest.approx(1000.0)
    assert bucket.stats.penalties == 0


@pytest.mark.parametrize("rate", [0, -1, -0.5])
def test_token_bucket_rejects_non_positive_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TokenBucket(rate)


def test_stats_is_a_snapshot_not_a_live_reference() -> None:
    bucket = TokenBucket(1000)
    before = bucket.stats

    bucket.acquire()
    bucket.penalize()

    assert before.permits == 0
    assert before.penalties == 0
    assert bucket.stats.permits == 1
    assert bucket.stats.penalties == 1

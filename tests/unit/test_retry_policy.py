from __future__ import annotations

import random
from datetime import UTC, datetime
from email.utils import format_datetime

import pytest

from agent.retry_policy import CircuitBreaker, backoff, parse_retry_after


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_parse_retry_after_seconds() -> None:
    now = datetime.now(UTC)
    assert parse_retry_after("0", now) == 0.0
    assert parse_retry_after("5", now) == 5.0
    assert parse_retry_after("  120 ", now) == 120.0


def test_parse_retry_after_date() -> None:
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    when = datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
    assert parse_retry_after(format_datetime(when, usegmt=True), now) == pytest.approx(30.0)


def test_parse_retry_after_past_date_is_zero() -> None:
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    past = datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC)
    assert parse_retry_after(format_datetime(past, usegmt=True), now) == 0.0


def test_parse_retry_after_naive_now() -> None:
    now = datetime(2024, 1, 1, 12, 0, 0)
    when = datetime(2024, 1, 1, 12, 0, 10, tzinfo=UTC)
    assert parse_retry_after(format_datetime(when, usegmt=True), now) == pytest.approx(10.0)


def test_parse_retry_after_none_or_garbage() -> None:
    now = datetime.now(UTC)
    assert parse_retry_after(None, now) is None
    assert parse_retry_after("", now) is None
    assert parse_retry_after("   ", now) is None
    assert parse_retry_after("not-a-date", now) is None
    assert parse_retry_after("-5", now) is None


def test_backoff_deterministic_jitter() -> None:
    for attempt in range(5):
        value = 1.0 * (2**attempt)
        expected = min(random.Random(0).uniform(0.0, value), 12.0)
        assert backoff(attempt, rng=random.Random(0)) == expected


def test_backoff_respects_cap() -> None:
    for seed in range(10):
        assert backoff(10, rng=random.Random(seed)) <= 12.0
        assert backoff(30, rng=random.Random(seed)) <= 12.0
    assert backoff(0, rng=random.Random(0)) >= 0.0


def test_backoff_with_retry_after() -> None:
    # retry_after larger than cap wins outright.
    assert backoff(0, retry_after=40.0, rng=random.Random(0)) == 40.0
    # otherwise retry_after and jitter compete, bounded by max(cap, retry_after).
    expected = max(2.0, random.Random(0).uniform(0.0, 8.0))
    assert backoff(3, retry_after=2.0, rng=random.Random(0)) == expected
    assert backoff(3, retry_after=2.0, rng=random.Random(0)) <= max(12.0, 2.0)


def test_breaker_default_threshold() -> None:
    breaker = CircuitBreaker(clock=FakeClock())
    assert breaker.state == "closed"
    assert breaker.allow() is True
    for _ in range(5):
        breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_breaker_success_resets_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
    breaker.record_failure()
    breaker.record_success()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"


def test_breaker_open_until_time_elapses() -> None:
    clock = FakeClock(start=100.0)
    breaker = CircuitBreaker(failure_threshold=1, open_seconds=30.0, clock=clock)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False
    clock.advance(29.0)
    assert breaker.allow() is False
    assert breaker.state == "open"
    clock.advance(1.0)
    assert breaker.allow() is True
    assert breaker.state == "half_open"


def test_breaker_single_trial_call() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, open_seconds=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(10.0)
    assert breaker.allow() is True
    assert breaker.state == "half_open"
    assert breaker.allow() is False
    assert breaker.allow() is False
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_breaker_half_open_failure_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, open_seconds=10.0, clock=clock)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(10.0)
    assert breaker.allow() is True
    assert breaker.state == "half_open"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_breaker_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0, clock=FakeClock())

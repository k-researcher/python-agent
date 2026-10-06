"""Retry and circuit-breaker policy for provider calls."""

from __future__ import annotations

import email.utils
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

CircuitState = Literal["closed", "open", "half_open"]


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Parse a Retry-After header into seconds to wait (>= 0) or None."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        seconds = None
    if seconds is not None and seconds >= 0:
        return seconds
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (when - now).total_seconds())


def backoff(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 12.0,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Return seconds to wait before the next attempt (exponential full jitter)."""
    value = base * (2**attempt)
    jitter = rng.uniform(0.0, value) if rng is not None else random.uniform(0.0, value)
    jitter = min(jitter, cap)
    if retry_after is None:
        return jitter
    return min(max(retry_after, jitter), max(cap, retry_after))


class CircuitBreaker:
    """Track consecutive provider failures and temporarily stop calls while open."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        open_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._clock = clock
        self._state: CircuitState = "closed"
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._trial_used = False

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit."""
        return self._state

    def allow(self) -> bool:
        """Return true when a call may be issued right now."""
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._opened_at is not None and (
                self._clock() - self._opened_at >= self._open_seconds
            ):
                self._state = "half_open"
                self._trial_used = True
                return True
            return False
        if self._trial_used:
            return False
        self._trial_used = True
        return True

    def record_success(self) -> None:
        """Reset the circuit to closed after a successful call."""
        self._state = "closed"
        self._consecutive_failures = 0
        self._trial_used = False
        self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call and open the circuit when needed."""
        if self._state == "closed":
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open()
        elif self._state == "half_open":
            self._open()

    def _open(self) -> None:
        self._state = "open"
        self._opened_at = self._clock()
        self._trial_used = False

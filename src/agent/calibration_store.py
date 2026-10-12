"""Persist token calibration in the database and update it with compare-and-swap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.models import TokenCalibration, utcnow
from agent.token_calibration import ALPHA, MAX_FACTOR, MIN_FACTOR, CalibrationKey


@dataclass(frozen=True, slots=True)
class Calibration:
    """Stored correction factor and the number of observations behind it."""

    factor: float
    samples: int


def _next_factor(old: float, raw_estimate: int, prompt_tokens: int) -> float:
    """Return the EWMA factor after one observation, clamped to the allowed range."""
    ratio = prompt_tokens / raw_estimate
    value = (1.0 - ALPHA) * old + ALPHA * ratio
    return min(max(value, MIN_FACTOR), MAX_FACTOR)


async def load_calibration(db: AsyncSession, key: CalibrationKey) -> Calibration:
    """Read the stored calibration for a key. Default to an identity factor."""
    row = await _read_calibration_row(db, key)
    if row is None:
        return Calibration(factor=1.0, samples=0)
    return Calibration(factor=row.factor, samples=row.samples)


async def _read_calibration_row(db: AsyncSession, key: CalibrationKey) -> TokenCalibration | None:
    """Return one calibration row for the key, or None when it is absent."""
    statement = (
        select(TokenCalibration)
        .where(
            TokenCalibration.provider == key.provider,
            TokenCalibration.model == key.model,
            TokenCalibration.wire_version == key.wire_version,
        )
        .execution_options(populate_existing=True)
    )
    return (await db.execute(statement)).scalar_one_or_none()


async def record_observation(
    db: AsyncSession,
    key: CalibrationKey,
    raw_estimate: int,
    prompt_tokens: int | None,
    *,
    max_attempts: int = 5,
) -> Calibration | None:
    """Apply one observation to the persisted calibration.

    Invalid observations return None and leave the database untouched. Each
    attempt runs in its own short transaction. A compare-and-swap on ``samples``
    detects concurrent writes; a conflict triggers a re-read and a retry.
    """
    if raw_estimate <= 0 or prompt_tokens is None or prompt_tokens <= 0:
        return None

    for _ in range(max_attempts):
        result = await _attempt(db, key, raw_estimate, prompt_tokens)
        if result is not None:
            return result
    return None


async def _attempt(
    db: AsyncSession,
    key: CalibrationKey,
    raw_estimate: int,
    prompt_tokens: int,
) -> Calibration | None:
    """Run one compare-and-swap attempt. Return None when it collides."""
    try:
        row = await _read_calibration_row(db, key)
        now = utcnow()
        if row is None:
            new_factor = _next_factor(1.0, raw_estimate, prompt_tokens)
            db.add(
                TokenCalibration(
                    provider=key.provider,
                    model=key.model,
                    wire_version=key.wire_version,
                    factor=new_factor,
                    samples=1,
                    updated_at=now,
                )
            )
            try:
                await db.commit()
            except IntegrityError:
                # Another process created the row first; retry as an update.
                await db.rollback()
                return None
            return Calibration(factor=new_factor, samples=1)

        new_factor = _next_factor(row.factor, raw_estimate, prompt_tokens)
        new_samples = row.samples + 1
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(TokenCalibration)
                .where(
                    TokenCalibration.provider == key.provider,
                    TokenCalibration.model == key.model,
                    TokenCalibration.wire_version == key.wire_version,
                    TokenCalibration.samples == row.samples,
                )
                .values(factor=new_factor, samples=new_samples, updated_at=now)
            ),
        )
        if result.rowcount == 0:
            # Another process changed the row between read and write.
            await db.rollback()
            return None
        await db.commit()
        return Calibration(factor=new_factor, samples=new_samples)
    except BaseException:
        # asyncio.CancelledError is not an Exception subclass: roll back the half-applied
        # UPDATE and re-raise instead of leaving the session in an open transaction.
        await db.rollback()
        raise

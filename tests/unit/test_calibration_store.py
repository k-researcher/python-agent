"""Tests for the persistent calibration store with compare-and-swap updates."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import Select, event, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import calibration_store
from agent.calibration_store import Calibration, load_calibration, record_observation
from agent.database import engine, init_database, session_factory
from agent.models import TokenCalibration, utcnow
from agent.token_calibration import MAX_FACTOR, MIN_FACTOR, CalibrationKey


def key(name: str) -> CalibrationKey:
    """Return a distinct calibration key per test to keep rows isolated."""
    return CalibrationKey("openai", f"test-{name}", "chat-json-v1")


async def _assert_cal(cal: Calibration | None, *, factor: float, samples: int) -> None:
    assert cal is not None
    assert cal.samples == samples
    assert cal.factor == pytest.approx(factor)


async def _read_row(db: AsyncSession, k: CalibrationKey) -> TokenCalibration | None:
    return await db.scalar(
        select(TokenCalibration).where(
            TokenCalibration.provider == k.provider,
            TokenCalibration.model == k.model,
            TokenCalibration.wire_version == k.wire_version,
        )
    )


async def test_load_calibration_defaults_to_identity() -> None:
    await init_database()
    async with session_factory() as db:
        assert await load_calibration(db, key("default")) == Calibration(1.0, 0)


async def test_load_calibration_reads_through_the_passed_session() -> None:
    await init_database()
    k = key("uncommitted")
    async with session_factory() as db:
        # A row added to but not yet committed in THIS session must be visible: a read
        # implementation that opened its own session would return the default (1.0, 0).
        db.add(
            TokenCalibration(
                provider=k.provider,
                model=k.model,
                wire_version=k.wire_version,
                factor=1.5,
                samples=7,
                updated_at=utcnow(),
            )
        )
        await _assert_cal(await load_calibration(db, k), factor=1.5, samples=7)


async def test_first_observation_creates_a_row() -> None:
    await init_database()
    k = key("first")
    async with session_factory() as db:
        result = await record_observation(db, k, 100, 200)
        await _assert_cal(result, factor=1.2, samples=1)
        stored = await load_calibration(db, k)
        await _assert_cal(stored, factor=1.2, samples=1)
    # A fresh session sees the committed row.
    async with session_factory() as db:
        row = await _read_row(db, k)
        assert row is not None
        assert row.factor == pytest.approx(1.2)
        assert row.samples == 1


async def test_ewma_formula_across_two_observations() -> None:
    await init_database()
    k = key("ewma")
    async with session_factory() as db:
        first = await record_observation(db, k, 100, 120)
        await _assert_cal(first, factor=0.8 * 1.0 + 0.2 * 1.2, samples=1)
        second = await record_observation(db, k, 100, 140)
        # Second factor builds on the first one: 0.8 * 1.04 + 0.2 * 1.4.
        await _assert_cal(second, factor=0.8 * 1.04 + 0.2 * 1.4, samples=2)


async def test_factor_is_clamped_to_both_edges() -> None:
    await init_database()
    k = key("clamp")
    async with session_factory() as db:
        for _ in range(50):
            result = await record_observation(db, k, 100, 1)
        await _assert_cal(result, factor=MIN_FACTOR, samples=50)
    async with session_factory() as db:
        for _ in range(50):
            result = await record_observation(db, k, 1, 1000)
        await _assert_cal(result, factor=MAX_FACTOR, samples=100)


async def test_invalid_observation_does_not_touch_the_row() -> None:
    await init_database()
    k = key("invalid")
    async with session_factory() as db:
        await record_observation(db, k, 100, 120)
    async with session_factory() as db:
        before = await load_calibration(db, k)
        assert await record_observation(db, k, 0, 120) is None
        assert await record_observation(db, k, -5, 120) is None
        assert await record_observation(db, k, 100, None) is None
        assert await record_observation(db, k, 100, 0) is None
        assert await record_observation(db, k, 100, -10) is None
        assert await load_calibration(db, k) == before


async def test_two_sequential_observations_give_two_samples() -> None:
    await init_database()
    k = key("sequential")
    async with session_factory() as db:
        first = await record_observation(db, k, 100, 100)
        await _assert_cal(first, factor=1.0, samples=1)
        second = await record_observation(db, k, 100, 100)
        await _assert_cal(second, factor=1.0, samples=2)


async def test_cas_conflict_retries_through_passed_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await init_database()
    k = key("cas")
    async with session_factory() as db:
        first = await record_observation(db, k, 100, 140)
        await _assert_cal(first, factor=0.8 * 1.0 + 0.2 * 1.4, samples=1)

        real_read = calibration_store._read_calibration_row
        passed_reads = 0
        injected = False

        async def observe_concurrently() -> None:
            # Another process records an observation that changes BOTH factor and samples,
            # so a buggy re-read that reuses the stale factor would compute the wrong EWMA.
            async with session_factory() as db:
                await record_observation(db, k, 100, 200)

        async def racing_read(_db: AsyncSession, _key: CalibrationKey) -> TokenCalibration | None:
            nonlocal passed_reads, injected
            if _db is db:
                passed_reads += 1
                if not injected:
                    # Inject the concurrent write right after the FIRST read of the passed
                    # session: the stale row stays in its identity map (populate_existing
                    # refreshes it on the retry), while the observation runs separately.
                    injected = True
                    stale = await real_read(_db, k)
                    await observe_concurrently()
                    return stale
            return await real_read(_db, k)

        # Record every statement that ran on the passed session, to prove the SELECTs
        # (not only the UPDATEs) are executed through it rather than a fresh session.
        executed: list[Any] = []
        real_execute = db.execute

        async def spy_execute(stmt: Any, *args: Any, **kwargs: Any) -> Any:
            executed.append(stmt)
            return await real_execute(stmt, *args, **kwargs)

        monkeypatch.setattr(db, "execute", spy_execute)
        monkeypatch.setattr(calibration_store, "_read_calibration_row", racing_read)

        result = await record_observation(db, k, 100, 100)
        # The competitor's observation moved the row to (1.264, 2); the retry reads that
        # row and applies the EWMA on top of the fresh factor: 0.8 * 1.264 + 0.2 * 1.0.
        await _assert_cal(result, factor=0.8 * 1.264 + 0.2 * 1.0, samples=3)

        # Exactly two SELECTs ran, both through this session. A _read_calibration_row
        # that ignored db and opened its own session would leave only UPDATE statements
        # here, so this assertion is what makes such a regression fail.
        selects = [s for s in executed if isinstance(s, Select)]
        assert len(selects) == 2
        assert passed_reads == 2

    # The committed result is visible from a fresh session without duplicate rows.
    monkeypatch.setattr(calibration_store, "_read_calibration_row", real_read)
    async with session_factory() as db:
        stored = await load_calibration(db, k)
        await _assert_cal(stored, factor=0.8 * 1.264 + 0.2 * 1.0, samples=3)
        rows = (
            (await db.execute(select(TokenCalibration).where(TokenCalibration.model == k.model)))
            .scalars()
            .all()
        )
        assert len(rows) == 1


async def test_pk_conflict_on_insert_retries_as_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await init_database()
    k = key("insert-conflict")
    real_read = calibration_store._read_calibration_row
    read_calls = 0

    async def insert_races(_db: AsyncSession, _key: CalibrationKey) -> TokenCalibration | None:
        nonlocal read_calls
        read_calls += 1
        if read_calls == 1:
            # Another process wins the race and creates the row first.
            async with session_factory() as db:
                db.add(
                    TokenCalibration(
                        provider=k.provider,
                        model=k.model,
                        wire_version=k.wire_version,
                        factor=1.2,
                        samples=1,
                        updated_at=utcnow(),
                    )
                )
                await db.commit()
            # Our read saw no row, so the INSERT will hit the primary key conflict.
            return None
        return await real_read(_db, k)

    monkeypatch.setattr(calibration_store, "_read_calibration_row", insert_races)

    async with session_factory() as db:
        result = await record_observation(db, k, 100, 100)
    # The conflicting INSERT was retried as an update on top of factor=1.2.
    await _assert_cal(result, factor=0.8 * 1.2 + 0.2 * 1.0, samples=2)
    async with session_factory() as db:
        stored = await load_calibration(db, k)
        assert stored == result


async def test_different_keys_do_not_mix() -> None:
    await init_database()
    k_a = key("mix-a")
    k_b = key("mix-b")
    async with session_factory() as db:
        await record_observation(db, k_a, 100, 150)
        await record_observation(db, k_b, 100, 300)
    async with session_factory() as db:
        await _assert_cal(await load_calibration(db, k_a), factor=1.1, samples=1)
        await _assert_cal(await load_calibration(db, k_b), factor=1.4, samples=1)


async def test_exhausted_attempts_return_none(monkeypatch: pytest.MonkeyPatch) -> None:
    await init_database()
    k = key("exhausted")
    async with session_factory() as db:
        await record_observation(db, k, 100, 140)

    real_read = calibration_store._read_calibration_row
    attempts = 0

    async def always_conflicts(_db: AsyncSession, _key: CalibrationKey) -> TokenCalibration | None:
        # Every read returns a row with a samples value the database no longer has, so
        # every update misses and every attempt has to be retried.
        nonlocal attempts
        attempts += 1
        async with session_factory() as db:
            row = await real_read(db, k)
        assert row is not None
        row.samples += 1
        return row

    monkeypatch.setattr(calibration_store, "_read_calibration_row", always_conflicts)

    async with session_factory() as db:
        result = await record_observation(db, k, 100, 100, max_attempts=3)
    assert result is None
    # Every retry must actually re-read, so exactly max_attempts attempts ran.
    assert attempts == 3
    # The row keeps the original (1.08, 1): no failed attempt persisted anything.
    async with session_factory() as db:
        row = await _read_row(db, k)
        assert row is not None
        assert row.factor == pytest.approx(0.8 * 1.0 + 0.2 * 1.4)
        assert row.samples == 1


async def test_cancel_after_update_rolls_back_and_session_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await init_database()
    k = key("cancel")
    async with session_factory() as db:
        # Seed the row with (factor=1.0, samples=1).
        first = await record_observation(db, k, 100, 100)
        await _assert_cal(first, factor=1.0, samples=1)

        commit_hits = 0

        async def cancel_before_commit() -> None:
            # Cancellation lands after the UPDATE was executed but before its commit.
            nonlocal commit_hits
            commit_hits += 1
            raise asyncio.CancelledError()

        with monkeypatch.context() as m:
            m.setattr(db, "commit", cancel_before_commit)
            # From (1.0, 1) with raw_estimate=100, prompt_tokens=200 the UPDATE computes
            # (1.2, 2); cancellation must discard that pending write, not just swallow it.
            with pytest.raises(asyncio.CancelledError):
                await record_observation(db, k, 100, 200)
        assert commit_hits == 1

        # Rollback was performed: the session no longer sits in an open transaction.
        assert not db.in_transaction()

        # An explicit commit afterwards must not persist the cancelled observation.
        await db.commit()
        row = await _read_row(db, k)
        assert row is not None
        assert row.factor == pytest.approx(1.0)
        assert row.samples == 1

        # The same session is still usable: a fresh observation commits cleanly.
        second = await record_observation(db, k, 100, 200)
        await _assert_cal(second, factor=0.8 * 1.0 + 0.2 * 2.0, samples=2)


async def test_invalid_observation_issues_no_sql_flush_commit_or_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await init_database()
    k = key("invalid-quiet")
    async with session_factory() as db:
        await record_observation(db, k, 100, 120)

    # Count SQL at engine level so queries emitted through db.scalar, db.get or any
    # other session API are caught too, not just the ones via db.execute.
    sync_engine = engine.sync_engine
    sql_calls: list[str] = []

    def spy_before_cursor_execute(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        sql_calls.append(statement)

    event.listen(sync_engine, "before_cursor_execute", spy_before_cursor_execute)
    try:
        async with session_factory() as db:
            transaction_events: list[str] = []
            real_commit = db.commit
            real_rollback = db.rollback

            async def spy_commit() -> None:
                transaction_events.append("commit")
                await real_commit()

            async def spy_rollback() -> None:
                transaction_events.append("rollback")
                await real_rollback()

            def spy_before_flush(session: Any, flush_context: Any, instances: Any) -> None:
                # Detects automatic or explicit flushes even when they emit no SQL.
                transaction_events.append("flush")

            event.listen(db.sync_session, "before_flush", spy_before_flush)
            monkeypatch.setattr(db, "commit", spy_commit)
            monkeypatch.setattr(db, "rollback", spy_rollback)

            assert await record_observation(db, k, 0, 120) is None
            assert await record_observation(db, k, -5, 120) is None
            assert await record_observation(db, k, 100, None) is None
            assert await record_observation(db, k, 100, 0) is None
            assert await record_observation(db, k, 100, -10) is None

            # Invalid observations short-circuit before touching the database: no cursor
            # execution, no commit/rollback and no (even empty) flush.
            assert sql_calls == []
            assert transaction_events == []
    finally:
        event.remove(sync_engine, "before_cursor_execute", spy_before_cursor_execute)


async def test_each_key_component_separates_rows() -> None:
    await init_database()
    base = key("key-parts")
    different_provider = CalibrationKey("other-provider", base.model, base.wire_version)
    different_version = CalibrationKey(base.provider, base.model, "chat-json-v2")
    async with session_factory() as db:
        await record_observation(db, base, 100, 150)
        await record_observation(db, different_provider, 100, 300)
        await record_observation(db, different_version, 100, 200)
    async with session_factory() as db:
        await _assert_cal(await load_calibration(db, base), factor=1.1, samples=1)
        await _assert_cal(await load_calibration(db, different_provider), factor=1.4, samples=1)
        await _assert_cal(await load_calibration(db, different_version), factor=1.2, samples=1)
    # The same model name at another provider or wire version is a separate physical row.
    async with session_factory() as db:
        rows = (
            (await db.execute(select(TokenCalibration).where(TokenCalibration.model == base.model)))
            .scalars()
            .all()
        )
        assert len(rows) == 3


async def test_update_of_one_key_does_not_touch_the_others() -> None:
    await init_database()
    base = key("update-isolation")
    diff_provider = CalibrationKey("other-provider", base.model, base.wire_version)
    diff_version = CalibrationKey(base.provider, base.model, "chat-json-v2")
    async with session_factory() as db:
        await record_observation(db, base, 100, 150)  # (1.1, 1)
        await record_observation(db, diff_provider, 100, 300)  # (1.4, 1)
        await record_observation(db, diff_version, 100, 200)  # (1.2, 1)
    # Re-observe only the base key so its row goes through the UPDATE path.
    async with session_factory() as db:
        result = await record_observation(db, base, 100, 220)
    # base: (1.1, 1) with ratio 2.2 -> 0.8 * 1.1 + 0.2 * 2.2, samples 2.
    await _assert_cal(result, factor=0.8 * 1.1 + 0.2 * 2.2, samples=2)
    # The UPDATE must be scoped by the full key: same-model rows at another provider or
    # wire version keep their original exact values.
    async with session_factory() as db:
        await _assert_cal(await load_calibration(db, base), factor=0.8 * 1.1 + 0.2 * 2.2, samples=2)
        await _assert_cal(await load_calibration(db, diff_provider), factor=1.4, samples=1)
        await _assert_cal(await load_calibration(db, diff_version), factor=1.2, samples=1)


async def test_exception_after_update_rolls_back_and_re_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await init_database()
    k = key("exception-rollback")
    async with session_factory() as db:
        first = await record_observation(db, k, 100, 100)
        await _assert_cal(first, factor=1.0, samples=1)

        async def commit_raises() -> None:
            raise RuntimeError("commit failed")

        with monkeypatch.context() as m:
            m.setattr(db, "commit", commit_raises)
            # The UPDATE computed (1.2, 2); a failure before the commit must discard it.
            with pytest.raises(RuntimeError, match="commit failed"):
                await record_observation(db, k, 100, 200)

        # Rollback was performed, so the session no longer sits in an open transaction.
        assert not db.in_transaction()
        # An explicit commit afterwards must not persist the failed observation.
        await db.commit()
        row = await _read_row(db, k)
        assert row is not None
        assert row.factor == pytest.approx(1.0)
        assert row.samples == 1
        # The same session remains usable for a fresh observation.
        second = await record_observation(db, k, 100, 200)
        await _assert_cal(second, factor=0.8 * 1.0 + 0.2 * 2.0, samples=2)

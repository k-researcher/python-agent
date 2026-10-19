"""Durable budget state: store functions on top of the pure policy."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import OutboundAudit, Project, Session
from agent.session_budget import BudgetLimits
from agent.session_budget_store import (
    _aware_utc,
    budget_block_reason,
    budget_limits,
    load_budget_usage,
    mark_budget_stop,
    record_step,
)
from agent.session_service import create_session_record

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "max_inference_steps": 100,
        "max_session_tokens": None,
        "max_session_cost": None,
        "max_session_seconds": None,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def session_row(tmp_path: Path) -> tuple[str, Path]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name="budget-store", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id, title="budget-store", status="pending", created_at=NOW
        )
        db.add(session)
        await db.flush()
        await db.commit()
        return session.id, tmp_path


async def _with_budget_state(
    session_id: str,
    *,
    limits: dict[str, object] | None = None,
    steps: int | None = None,
    started_at: str | None = None,
    stop_reason: str | None = None,
    run_generation: int | None = None,
) -> None:
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        state: dict[str, object] = {}
        if limits is not None:
            state["limits"] = limits
        if steps is not None:
            state["steps"] = steps
        if started_at is not None:
            state["started_at"] = started_at
        if stop_reason is not None:
            state["stop_reason"] = stop_reason
        if run_generation is not None:
            state["run_generation"] = run_generation
            row.run_generation = run_generation
        row.budget_state = state
        await db.commit()


async def test_budget_limits_from_snapshot(session_row: tuple[str, Path]) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        limits={
            "max_steps": 5,
            "max_tokens": 1000,
            "max_cost": "1.25",
            "deadline_seconds": 30.0,
        },
    )
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        limits = budget_limits(_settings(max_session_tokens=50, max_session_cost="0.1"), row)
    assert limits == BudgetLimits(
        max_steps=5, max_tokens=1000, max_cost=Decimal("1.25"), deadline_seconds=30.0
    )


async def test_budget_limits_absent_snapshot_uses_server_defaults(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None and row.budget_state is None
        limits = budget_limits(
            _settings(
                max_session_tokens=5000,
                max_session_cost="4.20",
                max_session_seconds=120.0,
            ),
            row,
        )
    assert limits == BudgetLimits(
        max_steps=100, max_tokens=5000, max_cost=Decimal("4.20"), deadline_seconds=120.0
    )


async def test_budget_limits_partial_snapshot_fills_gaps(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(session_id, limits={"max_steps": 9})
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        limits = budget_limits(_settings(max_session_tokens=300, max_session_seconds=60.0), row)
    assert limits == BudgetLimits(max_steps=9, max_tokens=300, max_cost=None, deadline_seconds=60.0)


async def test_load_budget_usage_absent_state_is_zero_and_started_now(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        usage = await load_budget_usage(db, row, NOW)
    assert usage.steps == 0
    assert usage.known_tokens == 0
    assert usage.known_cost == Decimal("0")
    assert usage.has_unknown_tokens is False
    assert usage.has_unknown_cost is False
    assert usage.started_at == NOW


async def test_load_budget_usage_reads_state_and_ledger(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        run_generation=0,
        steps=4,
        started_at="2026-06-01T10:00:00+00:00",
    )
    async with session_factory() as db:
        db.add(
            OutboundAudit(
                session_id=session_id,
                category="llm",
                destination="https://llm.example",
                operation="chat:test",
                status="completed",
                prompt_tokens=10,
                completion_tokens=20,
                cost_amount="0.00003",
            )
        )
        await db.commit()
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        usage = await load_budget_usage(db, row, NOW)
    assert usage.steps == 4
    assert usage.known_tokens == 30
    assert usage.known_cost == Decimal("0.00003")
    assert usage.started_at == datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)


async def test_load_budget_usage_ledger_error_raises(
    session_row: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id, _ = session_row

    async def broken(db: object, session_id: str) -> object:
        raise RuntimeError("ledger down")

    monkeypatch.setattr("agent.session_budget_store.summarize_session_usage", broken)
    async with session_factory() as db:
        import agent.session_budget_store as store

        monkeypatch.setattr(store, "summarize_session_usage", broken)
        row = await db.get(Session, session_id)
        assert row is not None
        with pytest.raises(RuntimeError, match="ledger down"):
            await load_budget_usage(db, row, NOW)


async def test_record_step_atomic_increment_and_start_time(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    async with session_factory() as db:
        first = await record_step(db, session_id, generation=1)
        await db.flush()
        second = await record_step(db, session_id, generation=1)
        await db.commit()
        assert (first, second) == (1, 2)
    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None
        assert state["steps"] == 2
        assert "started_at" in state
        assert state["run_generation"] == 1


async def test_record_step_keeps_existing_start_time(session_row: tuple[str, Path]) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        run_generation=1,
        steps=0,
        started_at="2026-06-01T00:00:00+00:00",
    )
    async with session_factory() as db:
        value = await record_step(db, session_id, generation=1)
        await db.commit()
        assert value == 1
    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None
        assert state["started_at"] == "2026-06-01T00:00:00+00:00"
        assert state["steps"] == 1


async def test_record_step_new_generation_resets_counters_and_keeps_limits(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        limits={"max_steps": 9},
        run_generation=1,
        steps=8,
        started_at="2026-06-01T00:00:00+00:00",
        stop_reason="steps_exhausted",
    )
    async with session_factory() as db:
        value = await record_step(db, session_id, generation=2)
        await db.commit()
        assert value == 1
    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None
        assert state["run_generation"] == 2
        assert state["steps"] == 1
        assert state["started_at"] != "2026-06-01T00:00:00+00:00"
        assert "stop_reason" not in state
        assert state["limits"] == {"max_steps": 9}


async def test_record_step_without_state_starts_run_generation(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    async with session_factory() as db:
        value = await record_step(db, session_id, generation=5)
        await db.commit()
        assert value == 1
    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None
        assert state["run_generation"] == 5
        assert state["steps"] == 1
        assert "started_at" in state


async def test_record_steps_in_separate_transactions_accumulate(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        limits={"max_steps": 9},
        run_generation=3,
        steps=0,
        started_at="2026-06-01T00:00:00+00:00",
    )

    for _ in range(3):
        async with session_factory() as db:
            await record_step(db, session_id, generation=3)
            await db.commit()

    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None and state["steps"] == 3
        assert state["limits"] == {"max_steps": 9}


async def test_mark_budget_stop_records_reason(session_row: tuple[str, Path]) -> None:
    session_id, _ = session_row
    await _with_budget_state(session_id, run_generation=1, steps=1)
    async with session_factory() as db:
        await mark_budget_stop(db, session_id, 1, "steps_exhausted")
        await db.commit()
    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None and state["stop_reason"] == "steps_exhausted"


async def test_mark_budget_stop_on_older_generation_is_ignored(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(session_id, run_generation=1, steps=1)
    async with session_factory() as db:
        await mark_budget_stop(db, session_id, 2, "steps_exhausted")
        await db.commit()
    async with session_factory() as db:
        state = await db.scalar(select(Session.budget_state).where(Session.id == session_id))
        assert state is not None and "stop_reason" not in state


async def test_budget_block_reason_when_allowed_and_blocked(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(session_id, run_generation=1, limits={"max_steps": 3}, steps=1)
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        assert await budget_block_reason(db, _settings(), row, NOW, new_run=False) is None
    await _with_budget_state(session_id, run_generation=1, limits={"max_steps": 3}, steps=3)
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        assert (
            await budget_block_reason(db, _settings(), row, NOW, new_run=False) == "steps_exhausted"
        )


async def test_budget_block_reason_new_run_ignores_steps_and_deadline(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    # The exhausted steps and an expired deadline belong to the current run; a new
    # run starts with fresh steps and time.
    await _with_budget_state(
        session_id,
        run_generation=1,
        limits={"max_steps": 3, "deadline_seconds": 1.0},
        steps=3,
        started_at="2026-06-01T00:00:00+00:00",
    )
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        assert await budget_block_reason(db, _settings(), row, NOW, new_run=True) is None


async def test_budget_block_reason_new_run_keeps_money_limit(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        run_generation=1,
        limits={"max_steps": 100, "max_cost": "0.01"},
        steps=1,
    )
    async with session_factory() as db:
        db.add(
            OutboundAudit(
                session_id=session_id,
                category="llm",
                destination="https://llm.example",
                operation="chat:test",
                status="completed",
                prompt_tokens=10,
                completion_tokens=20,
                cost_amount="0.05",
            )
        )
        await db.commit()
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        assert (
            await budget_block_reason(db, _settings(), row, NOW, new_run=True) == "cost_exhausted"
        )


async def test_budget_block_reason_older_generation_state_ignored(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    # The state is stamped with run_generation 1 but the session is already at 2:
    # steps and started_at of the old run must not apply.
    await _with_budget_state(
        session_id,
        run_generation=1,
        limits={"max_steps": 2, "deadline_seconds": 1.0},
        steps=2,
        started_at="2026-06-01T00:00:00+00:00",
    )
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        row.run_generation = 2
        await db.commit()
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        assert await budget_block_reason(db, _settings(), row, NOW, new_run=False) is None


async def test_budget_block_reason_deadline_and_unknown_cost(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        run_generation=1,
        limits={"max_cost": "1.00"},
        started_at="2026-06-01T11:00:00+00:00",
    )
    async with session_factory() as db:
        db.add(
            OutboundAudit(
                session_id=session_id,
                category="llm",
                destination="https://llm.example",
                operation="chat:test",
                status="completed",
                prompt_tokens=None,
                completion_tokens=None,
                cost_amount=None,
            )
        )
        await db.commit()
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        assert await budget_block_reason(db, _settings(), row, NOW, new_run=False) == "cost_unknown"


async def test_child_copies_parent_limits_without_spend(
    session_row: tuple[str, Path], tmp_path: Path
) -> None:
    parent_id, _ = session_row
    await _with_budget_state(
        parent_id,
        limits={"max_steps": 4, "max_tokens": 700, "max_cost": "0.50"},
        steps=3,
        started_at="2026-06-01T08:00:00+00:00",
    )
    async with session_factory() as db:
        parent = await db.get(Session, parent_id)
        assert parent is not None
        project = await db.get(Project, parent.project_id)
        assert project is not None
        child = await create_session_record(
            db,
            _settings(max_session_tokens=99),
            project,
            "child task",
            "dev",
            parent_id=parent.id,
        )
        child_state = await db.get(Session, child.id)
        assert child_state is not None and child_state.budget_state is not None
        state = child_state.budget_state
        expected_limits = {
            "max_steps": 4,
            "max_tokens": 700,
            "max_cost": "0.50",
            "deadline_seconds": None,
        }
        assert state["limits"] == expected_limits
        assert "steps" not in state or state["steps"] == 0
        limits = budget_limits(_settings(max_session_tokens=99), child_state)
        assert limits.max_steps == 4
        assert limits.max_tokens == 700
        assert limits.max_cost == Decimal("0.50")


async def test_fork_without_budget_state_enforces_server_values(
    session_row: tuple[str, Path], tmp_path: Path
) -> None:
    parent_id, _ = session_row
    await _with_budget_state(
        parent_id,
        limits={"max_steps": 2, "max_tokens": 99},
    )
    async with session_factory() as db:
        parent = await db.get(Session, parent_id)
        assert parent is not None
        # A fork copies the configuration but not the budget state: emulate the
        # message-tree fork by creating a fresh session with copied configuration.
        fork = Session(
            project_id=parent.project_id,
            mode="dev",
            llm_profile="default",
            title="fork",
            status="completed",
            configuration=dict(parent.configuration or {}),
            budget_state=None,
        )
        db.add(fork)
        await db.commit()
        await db.refresh(fork)
        assert fork.budget_state is None
        limits = budget_limits(_settings(max_session_tokens=123, max_inference_steps=200), fork)
        assert limits.max_steps == 200
        assert limits.max_tokens == 123


async def test_load_budget_usage_older_generation_ignores_steps_and_start(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        run_generation=1,
        limits={"max_steps": 3},
        steps=3,
        started_at="2026-06-01T00:00:00+00:00",
    )
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        row.run_generation = 2
        await db.commit()
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        usage = await load_budget_usage(db, row, NOW)
    assert usage.steps == 0
    assert usage.started_at == NOW
    assert usage.known_tokens == 0
    assert usage.known_cost == Decimal("0")


async def test_load_budget_usage_unparseable_started_at_is_none(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _with_budget_state(
        session_id,
        run_generation=0,
        steps=1,
        started_at="not-a-timestamp",
    )
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        usage = await load_budget_usage(db, row, NOW)
    assert usage.steps == 1
    assert usage.started_at is None


def test_aware_utc_unknown_value_returns_none() -> None:
    assert _aware_utc("not-a-timestamp") is None
    assert _aware_utc(None) is None
    assert _aware_utc(42) is None


class _RecordingDb:
    """Record the statements of a store function instead of running them."""

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> Any:
        self.statements.append(statement)
        row = SimpleNamespace(budget_state={"run_generation": 1, "steps": 1}, run_generation=1)
        return SimpleNamespace(one=lambda: row)


async def test_record_step_and_mark_budget_stop_compile_for_postgresql() -> None:
    """The store writes lock the row and use no DB-specific JSON functions."""
    db = _RecordingDb()
    await record_step(cast(AsyncSession, db), "s1", 1)
    await mark_budget_stop(cast(AsyncSession, db), "s1", 1, "steps_exhausted")
    compiled = [str(item.compile(dialect=postgresql.dialect())) for item in db.statements]
    assert len(compiled) == 4
    assert all("json_set" not in sql and "json_extract" not in sql for sql in compiled)
    assert "FOR UPDATE" in compiled[0] and "FOR UPDATE" in compiled[2]

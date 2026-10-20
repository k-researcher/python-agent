"""Durable checks of a single HTTP attempt against the session budget.

``check_attempt_budget`` decides before one HTTP attempt whether the worst case of the
attempt (request bytes as input tokens plus the requested output bound) and its cost fit
the session ceilings. These tests call the store function directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.model_config import PricingSpec
from agent.models import OutboundAudit, Project, Session
from agent.session_budget_store import BudgetExceeded, check_attempt_budget

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
        project = Project(name="attempt-budget", root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="attempt", status="running")
        db.add(session)
        await db.commit()
        return session.id, tmp_path


async def _run_check(
    session_id: str,
    settings: Settings,
    *,
    request_bytes: int,
    max_output_tokens: int,
    pricing: PricingSpec | None,
) -> None:
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        await check_attempt_budget(
            db,
            settings,
            session,
            request_bytes=request_bytes,
            max_output_tokens=max_output_tokens,
            pricing=pricing,
            now=NOW,
        )


async def _add_usage(session_id: str, *, tokens: int | None, cost: str | None) -> None:
    async with session_factory() as db:
        db.add(
            OutboundAudit(
                session_id=session_id,
                category="llm",
                destination="https://llm.example",
                operation="chat:test",
                status="completed",
                prompt_tokens=tokens,
                completion_tokens=tokens,
                cost_amount=cost,
            )
        )
        await db.commit()


async def test_allows_attempt_that_fits(session_row: tuple[str, Path]) -> None:
    session_id, _ = session_row
    settings = _settings(max_session_tokens=10_000, max_session_cost="1.00")
    await _run_check(
        session_id,
        settings,
        request_bytes=5_000,
        max_output_tokens=500,
        pricing=PricingSpec(input_per_mtok=1, output_per_mtok=1),
    )


async def test_refuses_attempt_when_known_tokens_plus_bound_exceed(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _add_usage(session_id, tokens=2_000, cost="0.01")
    # Known tokens (4 000) fit the ceiling alone; the worst case of the next attempt
    # (request bytes as input plus the output bound) pushes it over.
    settings = _settings(max_session_tokens=5_000)
    with pytest.raises(BudgetExceeded) as caught:
        await _run_check(
            session_id,
            settings,
            request_bytes=5_000,
            max_output_tokens=2_000,
            pricing=None,
        )
    assert caught.value.reason == "tokens_exhausted"


async def test_refuses_money_limit_without_pricing_as_cost_unknown(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    settings = _settings(max_session_cost="1.00")
    with pytest.raises(BudgetExceeded) as caught:
        await _run_check(
            session_id,
            settings,
            request_bytes=5_000,
            max_output_tokens=500,
            pricing=None,
        )
    assert caught.value.reason == "cost_unknown"


async def test_refuses_when_known_cost_plus_bound_cost_exceeds(
    session_row: tuple[str, Path],
) -> None:
    session_id, _ = session_row
    await _add_usage(session_id, tokens=1_000, cost="0.90")
    settings = _settings(max_session_cost="1.00")
    pricing = PricingSpec(input_per_mtok=1000, output_per_mtok=1000)
    with pytest.raises(BudgetExceeded) as caught:
        await _run_check(
            session_id,
            settings,
            request_bytes=5_000,
            max_output_tokens=500,
            pricing=pricing,
        )
    assert caught.value.reason == "cost_exhausted"


async def test_refuses_when_cost_ceiling_already_spent(session_row: tuple[str, Path]) -> None:
    session_id, _ = session_row
    await _add_usage(session_id, tokens=50, cost="0.90")
    # The cost ceiling alone (even a zero-cost boundary) already refuses the attempt.
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        with pytest.raises(BudgetExceeded) as caught:
            await check_attempt_budget(
                db,
                _settings(max_session_cost="0.50"),
                session,
                request_bytes=5_000,
                max_output_tokens=500,
                pricing=PricingSpec(input_per_mtok=0, output_per_mtok=0),
                now=NOW,
            )
        assert caught.value.reason == "cost_exhausted"


async def test_never_consumes_step_quota(session_row: tuple[str, Path]) -> None:
    """An attempt check must ignore the step counter: tools of the last allowed step run."""
    session_id, _ = session_row
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        row.budget_state = {
            "limits": {"max_steps": 1},
            "run_generation": row.run_generation or 0,
            "steps": 999,
            "started_at": NOW.isoformat(),
        }
        await db.commit()
    # No step or money limit: the exhausted step counter of the run must not block a
    # tool-level boundary.
    await _run_check(
        session_id,
        _settings(max_inference_steps=1),
        request_bytes=500,
        max_output_tokens=100,
        pricing=None,
    )


async def test_refuses_attempt_after_the_run_deadline(session_row: tuple[str, Path]) -> None:
    session_id, _ = session_row
    async with session_factory() as db:
        row = await db.get(Session, session_id)
        assert row is not None
        row.budget_state = {
            "run_generation": row.run_generation or 0,
            "steps": 1,
            "started_at": datetime(2026, 6, 1, 11, 0, 0, tzinfo=UTC).isoformat(),
        }
        await db.commit()
    with pytest.raises(BudgetExceeded) as caught:
        await _run_check(
            session_id,
            _settings(max_session_seconds=60),
            request_bytes=500,
            max_output_tokens=100,
            pricing=None,
        )
    assert caught.value.reason == "deadline_exceeded"

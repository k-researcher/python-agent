from __future__ import annotations

import asyncio
import json
from dataclasses import is_dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import JSON, String, select

from agent.attempt_audit import cancel_attempt, finish_attempt, start_attempt
from agent.context_inference import ContextInference
from agent.database import init_database, session_factory
from agent.llm import AttemptRecord, LLMClient
from agent.model_config import PricingSpec, ReasoningSpec, ResolvedModel, ResolvedModelRegistry
from agent.models import OutboundAudit, Project, Session, TokenCalibration
from agent.reasoning import ReasoningDecision
from agent.token_calibration import CalibrationKey


@pytest.fixture
async def audit_session(tmp_path: Path) -> tuple[str, CalibrationKey]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        project = Project(name=tmp_path.name, root_path=str(tmp_path))
        db.add(project)
        await db.flush()
        session = Session(project_id=project.id, title="attempt-cost", status="running")
        db.add(session)
        await db.commit()
        return session.id, CalibrationKey("test", session.id, "wire-cost-test")


async def _start(
    audit_session: tuple[str, CalibrationKey],
    pricing: PricingSpec | None,
    detail: str | None = None,
) -> tuple[int, AttemptRecord]:
    session_id, key = audit_session
    record = AttemptRecord(0, "started", 321, "a" * 64)
    audit_id = await start_attempt(
        session_id,
        {
            "destination": "https://llm.example/v1/chat/completions",
            "operation": "chat:test",
            "inference_id": session_id,
            "purpose": "inference",
            "provider": key.provider,
            "model_id": "test",
            "model": key.model,
            "wire_version": key.wire_version,
            "raw_estimate": 100,
            "factor": 1.0,
            **({"detail": detail} if detail is not None else {}),
        },
        record,
        pricing=pricing,
    )
    return audit_id, record


async def _row(audit_id: int) -> OutboundAudit:
    async with session_factory() as db:
        row = await db.get(OutboundAudit, audit_id)
        assert row is not None
        return row


def _assert_summary(
    summary: Any,
    *,
    attempts: int,
    tokens: int,
    cost: str,
    unknown_tokens: bool = False,
    unknown_cost: bool = False,
) -> None:
    from agent.usage_ledger import SessionUsageSummary

    assert isinstance(summary, SessionUsageSummary)
    assert is_dataclass(summary)
    assert type(summary.attempts) is int
    assert type(summary.known_tokens) is int
    assert isinstance(summary.known_cost, Decimal)
    assert summary.attempts == attempts
    assert summary.known_tokens == tokens
    assert summary.known_cost == Decimal(cost)
    assert summary.has_unknown_tokens is unknown_tokens
    assert summary.has_unknown_cost is unknown_cost


async def test_pricing_columns_are_nullable_json_and_string(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    snapshot = OutboundAudit.__table__.c.pricing_snapshot
    amount = OutboundAudit.__table__.c.cost_amount
    assert isinstance(snapshot.type, JSON) and snapshot.nullable
    assert isinstance(amount.type, String) and amount.type.length == 100 and amount.nullable
    audit_id, _ = await _start(audit_session, None)
    row = await _row(audit_id)
    assert row.pricing_snapshot is None
    assert row.cost_amount is None


async def test_omitted_pricing_remains_backwards_compatible(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    session_id, key = audit_session
    initial = AttemptRecord(0, "started", 0, "b" * 64)
    audit_id = await start_attempt(
        session_id, {"destination": "https://llm.example", "operation": "chat:test"}, initial
    )
    await finish_attempt(
        audit_id, replace(initial, status="completed", prompt_tokens=2, completion_tokens=3), key
    )
    row = await _row(audit_id)
    assert row.pricing_snapshot is None
    assert row.cost_amount is None
    assert (row.prompt_tokens, row.completion_tokens) == (2, 3)


@pytest.mark.parametrize(
    "input_rate,output_rate,prompt,completion,expected",
    [
        (0.1, 0.2, 1, 1, "0.0000003"),
        (0.123456789, 0.987654321, 3, 7, "0.000007283950614"),
        (1.25, 2.5, 1234567, 7654321, "20.67901125"),
        (0, 0, 12, 34, "0"),
        (0, 2, 10, 3, "0.000006"),
        (2, 0, 3, 10, "0.000006"),
        (1, 2, 0, 0, "0"),
        (1, 2, 0, 3, "0.000006"),
        (1, 2, 3, 0, "0.000003"),
    ],
)
async def test_finish_persists_exact_unrounded_decimal_string(
    audit_session: tuple[str, CalibrationKey],
    input_rate: float,
    output_rate: float,
    prompt: int,
    completion: int,
    expected: str,
) -> None:
    pricing = PricingSpec(input_per_mtok=input_rate, output_per_mtok=output_rate)
    audit_id, initial = await _start(audit_session, pricing)
    started = await _row(audit_id)
    assert started.pricing_snapshot == pricing.model_dump()
    assert started.cost_amount is None
    await finish_attempt(
        audit_id,
        replace(initial, status="completed", prompt_tokens=prompt, completion_tokens=completion),
        audit_session[1],
    )
    row = await _row(audit_id)
    assert isinstance(row.cost_amount, str)
    assert Decimal(row.cost_amount) == Decimal(expected)
    assert row.pricing_snapshot == pricing.model_dump()


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("invalid", [None, -1, True, False, 1.5, "7"])
async def test_finish_rejects_missing_or_invalid_token_counts(
    audit_session: tuple[str, CalibrationKey], field: str, invalid: Any
) -> None:
    audit_id, initial = await _start(
        audit_session, PricingSpec(input_per_mtok=0, output_per_mtok=0)
    )
    counts = {"prompt_tokens": 2, "completion_tokens": 3, field: invalid}
    terminal = replace(initial, status="error", error_kind="network", **counts)
    await finish_attempt(audit_id, terminal, audit_session[1])
    row = await _row(audit_id)
    assert row.status == "error"
    assert row.cost_amount is None


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"input_per_mtok": 1},
        {"output_per_mtok": 1},
        {"input_per_mtok": None, "output_per_mtok": 1},
        {"input_per_mtok": 1, "output_per_mtok": None},
        {"input_per_mtok": -1, "output_per_mtok": 1},
        {"input_per_mtok": 1, "output_per_mtok": -1},
        {"input_per_mtok": "not-a-rate", "output_per_mtok": 1},
        {"input_per_mtok": 1, "output_per_mtok": "not-a-rate"},
    ],
)
async def test_finish_with_missing_or_invalid_persisted_rates_has_unknown_cost(
    audit_session: tuple[str, CalibrationKey], snapshot: dict[str, Any] | None
) -> None:
    audit_id, initial = await _start(audit_session, None)
    async with session_factory() as db:
        row = await db.get(OutboundAudit, audit_id)
        assert row is not None
        row.pricing_snapshot = snapshot
        await db.commit()
    await finish_attempt(
        audit_id,
        replace(initial, status="completed", prompt_tokens=2, completion_tokens=3),
        audit_session[1],
    )
    row = await _row(audit_id)
    assert row.cost_amount is None
    assert row.pricing_snapshot == snapshot


async def test_snapshot_survives_original_pricing_mutation(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    pricing = PricingSpec(input_per_mtok=0.1, output_per_mtok=0.2)
    snapshot = pricing.model_dump()
    audit_id, initial = await _start(audit_session, pricing)
    object.__setattr__(pricing, "input_per_mtok", 999)
    object.__setattr__(pricing, "output_per_mtok", 888)
    await finish_attempt(
        audit_id,
        replace(initial, status="completed", prompt_tokens=1, completion_tokens=1),
        audit_session[1],
    )
    row = await _row(audit_id)
    assert row.pricing_snapshot == snapshot
    assert Decimal(row.cost_amount) == Decimal("0.0000003")


async def test_concurrent_and_repeated_finish_charges_once(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import summarize_session_usage

    audit_id, initial = await _start(
        audit_session, PricingSpec(input_per_mtok=0.1, output_per_mtok=0.2)
    )
    terminal = replace(initial, status="completed", prompt_tokens=200, completion_tokens=3)
    await asyncio.gather(*(finish_attempt(audit_id, terminal, audit_session[1]) for _ in range(8)))
    first = await _row(audit_id)
    await finish_attempt(
        audit_id, replace(terminal, prompt_tokens=999, completion_tokens=999), audit_session[1]
    )
    repeated = await _row(audit_id)
    assert repeated.finished_at == first.finished_at
    assert repeated.pricing_snapshot == first.pricing_snapshot
    assert repeated.cost_amount == first.cost_amount
    assert Decimal(repeated.cost_amount) == Decimal("0.0000206")
    assert (repeated.prompt_tokens, repeated.completion_tokens) == (200, 3)
    async with session_factory() as db:
        key = audit_session[1]
        calibration = await db.get(TokenCalibration, (key.provider, key.model, key.wire_version))
        assert calibration is not None and calibration.samples == 1
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=1, tokens=203, cost="0.0000206")


async def test_repeated_finish_does_not_replace_unknown_usage(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    audit_id, initial = await _start(
        audit_session, PricingSpec(input_per_mtok=1, output_per_mtok=2)
    )
    await finish_attempt(
        audit_id, replace(initial, status="error", error_kind="network"), audit_session[1]
    )
    first = await _row(audit_id)
    await finish_attempt(
        audit_id,
        replace(initial, status="completed", prompt_tokens=10, completion_tokens=20),
        audit_session[1],
    )
    repeated = await _row(audit_id)
    assert repeated.status == "error" and repeated.error_kind == "network"
    assert repeated.finished_at == first.finished_at
    assert repeated.pricing_snapshot == first.pricing_snapshot
    assert repeated.prompt_tokens is None and repeated.completion_tokens is None
    assert repeated.cost_amount is None


async def test_concurrent_distinct_attempts_preserve_each_cost(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import summarize_session_usage

    pricing = PricingSpec(input_per_mtok=0.1, output_per_mtok=0.2)
    started = [await _start(audit_session, pricing) for _ in range(10)]
    await asyncio.gather(
        *(
            finish_attempt(
                audit_id,
                replace(initial, status="completed", prompt_tokens=100, completion_tokens=1),
                audit_session[1],
            )
            for audit_id, initial in started
        )
    )
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
        rows = list(
            await db.scalars(
                select(OutboundAudit).where(OutboundAudit.session_id == audit_session[0])
            )
        )
    assert len(rows) == 10
    assert all(Decimal(row.cost_amount) == Decimal("0.0000102") for row in rows)
    _assert_summary(summary, attempts=10, tokens=1010, cost="0.000102")


async def test_cancelled_attempt_cannot_be_charged_by_late_finish(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import summarize_session_usage

    audit_id, initial = await _start(
        audit_session, PricingSpec(input_per_mtok=1, output_per_mtok=2)
    )
    await cancel_attempt(audit_id)
    await finish_attempt(
        audit_id,
        replace(initial, status="completed", prompt_tokens=10, completion_tokens=20),
        audit_session[1],
    )
    row = await _row(audit_id)
    assert row.status == "cancelled" and row.cost_amount is None
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=1, tokens=0, cost="0", unknown_tokens=True, unknown_cost=True)


async def test_empty_summary_is_known_zero(audit_session: tuple[str, CalibrationKey]) -> None:
    from agent.usage_ledger import summarize_session_usage

    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=0, tokens=0, cost="0")


@pytest.mark.parametrize("amount", ["0", "0.000000000123456789"])
async def test_summary_cost_is_independent_of_token_and_pricing_availability(
    audit_session: tuple[str, CalibrationKey], amount: str
) -> None:
    from agent.usage_ledger import summarize_session_usage

    async with session_factory() as db:
        db.add(
            OutboundAudit(
                session_id=audit_session[0],
                category="llm",
                destination="https://llm.example",
                operation="chat:test",
                status="error",
                prompt_tokens=None,
                completion_tokens=3,
                pricing_snapshot=None,
                cost_amount=amount,
            )
        )
        await db.commit()
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=1, tokens=3, cost=amount, unknown_tokens=True)


@pytest.mark.parametrize("amount", ["-0.5", "NaN", "Infinity", "not-a-number"])
async def test_summary_treats_invalid_stored_cost_as_unknown(
    audit_session: tuple[str, CalibrationKey], amount: str
) -> None:
    from agent.usage_ledger import summarize_session_usage

    async with session_factory() as db:
        for cost in (amount, "0.25"):
            db.add(
                OutboundAudit(
                    session_id=audit_session[0],
                    category="llm",
                    destination="https://llm.example",
                    operation="chat:test",
                    status="completed",
                    prompt_tokens=1,
                    completion_tokens=1,
                    pricing_snapshot=None,
                    cost_amount=cost,
                )
            )
        await db.commit()
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=2, tokens=4, cost="0.25", unknown_cost=True)


@pytest.mark.parametrize("status", ["started", "completed", "error", "cancelled"])
@pytest.mark.parametrize(
    "prompt,completion,tokens,unknown",
    [
        (None, None, 0, True),
        (5, None, 5, True),
        (None, 7, 7, True),
        (0, 0, 0, False),
        (5, 7, 12, False),
        (-1, 7, 7, True),
        (5, -1, 5, True),
        (1.5, 7, 7, True),
    ],
)
async def test_summary_counts_partial_usage_in_every_status(
    audit_session: tuple[str, CalibrationKey],
    status: str,
    prompt: Any,
    completion: Any,
    tokens: int,
    unknown: bool,
) -> None:
    from agent.usage_ledger import summarize_session_usage

    async with session_factory() as db:
        db.add(
            OutboundAudit(
                session_id=audit_session[0],
                category="llm",
                destination="https://llm.example",
                operation="chat:test",
                status=status,
                prompt_tokens=prompt,
                completion_tokens=completion,
                pricing_snapshot=None,
                cost_amount=None,
            )
        )
        await db.commit()
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(
        summary, attempts=1, tokens=tokens, cost="0", unknown_tokens=unknown, unknown_cost=True
    )


async def test_summary_uses_only_persisted_amounts_and_isolates_sessions_and_mcp(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import summarize_session_usage

    session_id, _ = audit_session
    async with session_factory() as db:
        session = await db.get(Session, session_id)
        assert session is not None
        other = Session(project_id=session.project_id, title="other", status="running")
        db.add(other)
        await db.flush()
        entries = [
            (session_id, "llm", "completed", 1, 1, "0.0000003"),
            (session_id, "llm", "error", 2, 3, "0.0000007"),
            (session_id, "llm", "cancelled", 0, 0, "0"),
            (session_id, "llm", "started", 4, None, None),
            (session_id, "llm", "completed", 5, 6, None),
            (session_id, "mcp", "completed", 1000, 2000, "999"),
            (session_id, "mcp", "started", None, None, None),
            (other.id, "llm", "completed", 3000, 4000, "888"),
            (other.id, "llm", "started", None, None, None),
            (None, "llm", "completed", 5000, 6000, "777"),
        ]
        for owner, category, status, prompt, completion, amount in entries:
            db.add(
                OutboundAudit(
                    session_id=owner,
                    category=category,
                    status=status,
                    destination="https://example.test",
                    operation="test",
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    cost_amount=amount,
                    pricing_snapshot={"input_per_mtok": 999, "output_per_mtok": 999},
                )
            )
        await db.commit()
    async with session_factory() as db:
        first = await summarize_session_usage(db, session_id)
        _assert_summary(
            first,
            attempts=5,
            tokens=22,
            cost="0.000001",
            unknown_tokens=True,
            unknown_cost=True,
        )
    async with session_factory() as db:
        repeated = await summarize_session_usage(db, session_id)
        assert repeated == first
        rows = list(
            await db.scalars(
                select(OutboundAudit).where(
                    OutboundAudit.session_id == session_id, OutboundAudit.category == "llm"
                )
            )
        )
        for row in rows:
            row.pricing_snapshot = None
        await db.commit()
    async with session_factory() as db:
        assert await summarize_session_usage(db, session_id) == first


async def test_context_inference_passes_model_pricing_to_persisted_attempt(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    pricing = PricingSpec(input_per_mtok=0.1, output_per_mtok=0.2)
    model = ResolvedModel(
        id="cost-test",
        provider_id="test",
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test",
        configured=True,
        model=audit_session[1].model,
        context_window=8000,
        max_tokens=1000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(supported=False),
        pricing=pricing,
    )
    registry = ResolvedModelRegistry(
        models={model.id: model},
        default_id=model.id,
        fallback=[],
        roles={},
        warnings=[],
        source="yaml",
        checksum="cost-test",
        light_id=None,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        chunk = {
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\n".encode())

    client = LLMClient(model)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client._http = http
        inference = ContextInference(
            audit_session[0],
            0,
            [],
            registry,
            lambda _: client,
            lambda _: ReasoningDecision(None, None, "none", "model"),
        )
        result = await inference.run(model)
    assert result.response.content == "hi"
    async with session_factory() as db:
        rows = list(
            await db.scalars(
                select(OutboundAudit).where(
                    OutboundAudit.session_id == audit_session[0], OutboundAudit.category == "llm"
                )
            )
        )
    assert len(rows) == 1
    assert rows[0].pricing_snapshot == pricing.model_dump()
    assert isinstance(rows[0].cost_amount, str)
    assert Decimal(rows[0].cost_amount) == Decimal("0.0000003")
    assert json.loads(rows[0].detail)["max_output_tokens"] > 0


async def test_attempt_without_usage_counts_at_its_upper_bound(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import attempt_detail, summarize_session_usage

    pricing = PricingSpec(input_per_mtok=1, output_per_mtok=2)
    cancelled, _ = await _start(audit_session, pricing, attempt_detail(1000))
    await cancel_attempt(cancelled)
    timed_out, initial = await _start(audit_session, pricing, attempt_detail(1000))
    await finish_attempt(
        timed_out, replace(initial, status="error", error_kind="timeout"), audit_session[1]
    )
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    # Each attempt: 321 request bytes as input tokens plus 1000 output tokens.
    _assert_summary(summary, attempts=2, tokens=2 * 1321, cost="0.004642")


async def test_bounded_attempt_without_pricing_keeps_cost_unknown(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import attempt_detail, summarize_session_usage

    audit_id, initial = await _start(audit_session, None, attempt_detail(1000))
    await finish_attempt(
        audit_id, replace(initial, status="error", error_kind="timeout"), audit_session[1]
    )
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=1, tokens=1321, cost="0", unknown_cost=True)


async def test_completed_attempt_without_usage_stays_unknown(
    audit_session: tuple[str, CalibrationKey],
) -> None:
    from agent.usage_ledger import attempt_detail, summarize_session_usage

    pricing = PricingSpec(input_per_mtok=1, output_per_mtok=2)
    audit_id, initial = await _start(audit_session, pricing, attempt_detail(1000))
    await finish_attempt(audit_id, replace(initial, status="completed"), audit_session[1])
    async with session_factory() as db:
        summary = await summarize_session_usage(db, audit_session[0])
    _assert_summary(summary, attempts=1, tokens=0, cost="0", unknown_tokens=True, unknown_cost=True)

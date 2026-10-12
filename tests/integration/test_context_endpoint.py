from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from agent.context_budget import input_budget
from agent.database import session_factory
from agent.llm import LLMClient
from agent.model_registry import load_registry
from agent.models import Event, Message, OutboundAudit, TokenCalibration
from tests.integration.helpers import authed_client


@pytest.fixture(autouse=True)
def isolated_context_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.api import settings

    monkeypatch.setattr(settings, "llm_model", f"context-{tmp_path.name}")
    monkeypatch.setattr(settings, "context_window", 32768)
    monkeypatch.setattr(settings, "context_reserved_tokens", 1024)
    monkeypatch.setattr(settings, "llm_max_tokens", 8192)

    async def registry_for_test(db, configuration):
        return load_registry(configuration)

    monkeypatch.setattr("agent.api.effective_registry", registry_for_test)


def _create_session(client, tmp_path: Path, prompt: str = "test") -> str:
    project = client.post("/api/projects", json={"name": "ctx", "root_path": str(tmp_path)})
    assert project.status_code == 201, project.text
    session = client.post(
        "/api/sessions",
        json={
            "project_id": project.json()["id"],
            "prompt": prompt,
            "mode": "ask",
            "auto_start": False,
        },
    )
    assert session.status_code == 201, session.text
    return session.json()["id"]


async def _add_exchanges(session_id: str, count: int, chunk: int) -> None:
    async with session_factory() as db:
        rows = []
        for index in range(count):
            rows.append(
                Message(
                    session_id=session_id,
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {"path": f"/tmp/f{index}", "content": "x" * chunk}
                                ),
                            },
                        }
                    ],
                )
            )
            rows.append(
                Message(
                    session_id=session_id,
                    role="tool",
                    tool_call_id=f"call-{index}",
                    content=json.dumps({"ok": True, "data": "r" * chunk}),
                )
            )
        db.add_all(rows)
        await db.commit()


async def _counts(session_id: str) -> dict[str, int]:
    async with session_factory() as db:
        summaries = await db.scalar(
            select(func.count(Message.id)).where(
                Message.session_id == session_id, Message.kind == "summary"
            )
        )
        audits = await db.scalar(
            select(func.count(OutboundAudit.id)).where(OutboundAudit.session_id == session_id)
        )
        events = await db.scalar(select(func.count(Event.id)).where(Event.session_id == session_id))
        calibrations = await db.scalar(select(func.count()).select_from(TokenCalibration))
    return {
        "summaries": summaries or 0,
        "audits": audits or 0,
        "events": events or 0,
        "calibrations": calibrations or 0,
    }


def test_context_full_fields(tmp_path: Path) -> None:
    with authed_client() as client:
        session_id = _create_session(client, tmp_path)
        response = client.get(f"/api/sessions/{session_id}/context")
        assert response.status_code == 200, response.text
        payload = response.json()
        model = payload["model_id"]
        assert model == "default"
        assert payload["provider"] == "default"
        assert payload["wire_version"]
        assert payload["raw_tokens"] > 0
        assert payload["tokens_before"] > 0
        from agent.api import settings

        budget = input_budget(settings.context_window, settings.context_reserved_tokens, 8192)
        assert payload["input_budget"] == budget.input_tokens
        assert payload["context_window"] == settings.context_window
        assert payload["reserved_tokens"] == settings.context_window - budget.input_tokens
        assert payload["output_tokens"] == budget.output_tokens
        assert payload["safety_margin"] == budget.safety_margin
        assert payload["effective_max_tokens"] == 8192
        assert payload["target_tokens"] == budget.input_tokens
        assert payload["calibration_factor"] == 1.0
        assert payload["calibration_samples"] == 0
        assert payload["fits"] == (payload["approximate_tokens"] <= payload["input_budget"])
        assert payload["stage"] == "none"
        assert payload["summary_used"] is False
        assert payload["summary_calls"] == 0
        assert payload["omitted_messages"] == 0
        assert payload["truncated_tool_results"] == 0
        assert payload["masked_tool_results"] == 0
        assert payload["folded_tool_calls"] == 0
        assert payload["summary_checkpoint_id"] is None
        assert payload["covers_until"] is None
        assert payload["would_summarize"] is False
        assert payload["error"] is None


def test_context_compressible_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbid(self: LLMClient, *args, **kwargs):
        raise AssertionError("LLMClient.send invoked during context preview")

    monkeypatch.setattr(LLMClient, "send", _forbid)
    with authed_client() as client:
        session_id = _create_session(client, tmp_path)
        client.portal.call(_add_exchanges, session_id, 60, 5000)
        before = client.portal.call(_counts, session_id)
        response = client.get(f"/api/sessions/{session_id}/context")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["fits"] is True
        assert payload["stage"] != "none" or payload["summary_used"] is True
        after = client.portal.call(_counts, session_id)
        assert after == before


def test_context_protected_overflow(tmp_path: Path) -> None:
    from agent.api import settings

    original_window = settings.context_window
    original_reserved = settings.context_reserved_tokens
    original_max_tokens = settings.llm_max_tokens
    try:
        settings.context_window = 4096
        settings.context_reserved_tokens = 1024
        settings.llm_max_tokens = 1024
        with authed_client() as client:
            session_id = _create_session(client, tmp_path)
            client.portal.call(_add_exchanges, session_id, 4, 3000)
            before = client.portal.call(_counts, session_id)
            response = client.get(f"/api/sessions/{session_id}/context")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["fits"] is False
            assert payload["error"] == "protected_over_budget"
            after = client.portal.call(_counts, session_id)
            assert after == before
    finally:
        settings.context_window = original_window
        settings.context_reserved_tokens = original_reserved
        settings.llm_max_tokens = original_max_tokens


def test_context_unknown_session_404(tmp_path: Path) -> None:
    with authed_client() as client:
        response = client.get("/api/sessions/does-not-exist/context")
        assert response.status_code == 404


def test_context_unauthenticated_401(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from agent.main import app

    with TestClient(app) as client:
        response = client.get("/api/sessions/any/context")
        assert response.status_code == 401

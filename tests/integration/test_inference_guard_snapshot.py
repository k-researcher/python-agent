"""All reload resources carry the same durable session watermark."""

import asyncio

import httpx
import pytest

from agent.api import settings
from agent.auth import issue_login_code
from agent.database import session_factory
from agent.main import app
from tests.inference_guard_helpers import guard_tasks as guard_tasks

pytestmark = pytest.mark.usefixtures("guard_tasks")


async def test_reload_resource_watermarks_and_guard_outcome(tmp_path):
    async with (
        asyncio.timeout(5),
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client,
    ):
        async with session_factory() as db:
            code = await issue_login_code(db, settings)
        login = await client.post(
            "/auth/login", json={"code": code}, headers={"Origin": "http://testserver"}
        )
        assert login.status_code == 200
        client.headers.update(
            {"Origin": "http://testserver", "X-Agent-CSRF": login.json()["csrf_token"]}
        )
        project = (
            await client.post(
                "/api/projects", json={"name": "snapshot", "root_path": str(tmp_path)}
            )
        ).json()
        response = await client.post(
            "/api/sessions",
            json={"project_id": project["id"], "prompt": "hello", "auto_start": False},
        )
        assert response.status_code == 201
        session_id = response.json()["id"]
        snapshots = [
            await client.get(f"/api/sessions/{session_id}/snapshot/{resource}")
            for resource in ("session", "messages", "approvals", "children", "audit", "context")
        ]
        assert all(item.status_code == 200 for item in snapshots)
        payloads = [item.json() for item in snapshots]
        assert len({item["revision"] for item in payloads}) == 1
        assert len({item["run_generation"] for item in payloads}) == 1
        assert all(item["consistent"] for item in payloads)
        assert all("data" in item for item in payloads)
        assert "error_code" in payloads[0]["data"]

        async def accept_stream():
            from agent.database import session_factory
            from agent.events import add_event
            from agent.models import Message

            async with session_factory() as db:
                message = Message(session_id=session_id, role="assistant", content="answer")
                db.add(message)
                await db.flush()
                await add_event(
                    db,
                    "message.created",
                    {
                        "role": "assistant",
                        "message_id": message.id,
                        "generation": response.json()["run_generation"],
                        "stream_no": 1,
                        "stream_id": "accepted",
                    },
                    session_id,
                )
                await db.commit()

        await accept_stream()
        accepted = (await client.get(f"/api/sessions/{session_id}/snapshot/messages")).json()
        assert accepted["completed_streams"] == [
            {
                "generation": response.json()["run_generation"],
                "stream_no": 1,
                "stream_id": "accepted",
            }
        ]

        async def fail_run():
            from agent.database import session_factory
            from agent.run_finish import finish_run

            async with session_factory() as db:
                await finish_run(
                    db,
                    session_id,
                    response.json()["run_generation"],
                    "error",
                    reason="llm_repetition",
                    error="Обнаружен повтор",
                )
                await db.commit()

        await fail_run()
        failed = (await client.get(f"/api/sessions/{session_id}/snapshot/session")).json()
        assert failed["data"]["error_code"] == "llm_repetition"
        assert failed["data"]["status"] == "error"
        assert failed["revision"] > payloads[0]["revision"]


async def test_audit_changes_advance_the_common_snapshot_watermark(tmp_path):
    async with asyncio.timeout(5):
        from agent.api import _snapshot_stamp
        from agent.attempt_audit import finish_attempt, start_attempt
        from agent.llm import AttemptRecord
        from agent.token_calibration import CalibrationKey
        from tests.integration.test_context_inference import _create_session

        session_id = await _create_session(tmp_path, "audit-revision")

        async def revision():
            async with session_factory() as db:
                return (await _snapshot_stamp(db, session_id))[1]

        before = await revision()
        audit_id = await start_attempt(
            session_id,
            {"destination": "https://example.test/v1", "operation": "chat:main"},
            AttemptRecord(0, "started", 1, "hash"),
        )
        started = await revision()
        assert started > before
        await finish_attempt(
            audit_id,
            AttemptRecord(0, "error", 1, "hash", 7, 3, "llm_repetition"),
            CalibrationKey("gateway", "engine", "chat"),
        )
        finished = await revision()
        assert finished > started
        await finish_attempt(
            audit_id,
            AttemptRecord(0, "error", 1, "hash"),
            CalibrationKey("gateway", "engine", "chat"),
        )
        assert await revision() == finished

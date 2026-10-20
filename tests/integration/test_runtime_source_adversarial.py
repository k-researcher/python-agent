"""Adversarial read-only MCP source-revision runtime tests.

These tests exercise the real supervisor run loop and fake two-turn LLM
responses to prove that a read-only MCP tool capture is guarded by its stored
source revision without any Approval, while a legacy builtin with a NULL
revision keeps running normally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.llm import LLMResponse
from agent.models import (
    Approval,
    Message,
    Project,
    Session,
    ToolCall,
    User,
    Workspace,
    WorkspaceMembership,
)
from agent.runtime import AgentSupervisor
from agent.tool_sources import ToolSource
from agent.tools.base import RiskLevel
from agent.tools.mcp_tools import McpBackedTool
from agent.tools.registry import ToolRegistry
from tests.integration.test_runtime_source_revision import (
    CountingBuiltin,
    CountingClient,
    add_call,
    bound_tool,
    load_session,
)


@dataclass
class ReadOnlyCase:
    session_id: str
    user_id: str
    source: ToolSource
    client: CountingClient
    tool: McpBackedTool
    supervisor: AgentSupervisor


class TwoTurnCallLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.turns = 0

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any
    ) -> LLMResponse:
        self.turns += 1
        if self.turns == 1:
            return LLMResponse(
                None,
                [
                    {
                        "id": f"adversarial_call_{self.turns}",
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": "{}"},
                    }
                ],
                "tool_calls",
                1,
                1,
            )
        return LLMResponse("completed", [], "stop", 1, 1)


class CompletionLLM:
    endpoint = "https://fake.example/v1/chat/completions"

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **options: Any
    ) -> LLMResponse:
        return LLMResponse("finished", [], "stop", 1, 1)


async def _seed_user_message(session_id: str, content: str = "Read") -> None:
    async with session_factory() as db:
        db.add(Message(session_id=session_id, role="user", content=content))
        await db.commit()


@pytest.fixture
async def read_only_case(tmp_path: Path) -> AsyncIterator[ReadOnlyCase]:
    await init_database(recover_running=False)
    async with session_factory() as db:
        user = User(display_name="Read-only actor")
        workspace = Workspace(name="Read-only workspace")
        db.add_all([user, workspace])
        await db.flush()
        project = Project(
            name="Read-only project", root_path=str(tmp_path), workspace_id=workspace.id
        )
        db.add_all(
            [
                project,
                WorkspaceMembership(user_id=user.id, workspace_id=workspace.id, role="member"),
            ]
        )
        await db.flush()
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Read-only run",
            status="running",
            configuration={},
        )
        db.add(session)
        await db.commit()
        source = ToolSource(
            source_id="bound",
            workspace_id=workspace.id,
            project_ids=frozenset({project.id}),
            command=("server",),
            enabled=True,
            risk_level=RiskLevel.read_only,
        )
        client = CountingClient()
        tool = bound_tool(client, source)
        supervisor = AgentSupervisor(Settings(), ToolRegistry(tools=(tool,)))
        yield ReadOnlyCase(session.id, user.id, source, client, tool, supervisor)


async def test_read_only_mcp_capture_persists_revision_and_executes_once(
    read_only_case: ReadOnlyCase,
) -> None:
    case = read_only_case
    case.supervisor.llm = TwoTurnCallLLM(case.tool.name)
    await _seed_user_message(case.session_id)
    await case.supervisor._run(case.session_id)
    assert case.client.calls == 1
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert session.status == "completed"
        assert len(session.tool_calls) == 1
        call = session.tool_calls[0]
        assert call.status == "completed"
        assert call.source_revision == case.tool.approval_revision()
        assert call.approval is None
        assert (await db.scalar(select(Approval).where(Approval.tool_call_id == call.id))) is None


async def test_read_only_pending_mcp_null_revision_denies_without_approval(
    read_only_case: ReadOnlyCase,
) -> None:
    case = read_only_case
    call_id = await add_call(case, revision=None, approval=None)
    case.supervisor.llm = CompletionLLM()
    await case.supervisor._run(case.session_id)
    assert case.client.calls == 0
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.status == "error"
        assert call.result is not None and call.result["success"] is False


async def test_legacy_builtin_null_revision_continues_normally(
    read_only_case: ReadOnlyCase,
) -> None:
    case = read_only_case
    legacy_tool = CountingBuiltin("builtin-legacy")
    case.supervisor.registry = ToolRegistry(tools=(legacy_tool,))
    case.supervisor.llm = TwoTurnCallLLM(legacy_tool.name)
    await _seed_user_message(case.session_id)
    await case.supervisor._run(case.session_id)
    assert legacy_tool.calls == 1
    async with session_factory() as db:
        session = await load_session(db, case.session_id)
        assert session.status == "completed"
        assert len(session.tool_calls) == 1
        call = session.tool_calls[0]
        assert call.name == legacy_tool.name
        assert call.status == "completed"
        assert call.source_revision is None
        assert call.approval is None


async def test_changed_schema_after_pending_capture_denies_without_approval(
    read_only_case: ReadOnlyCase,
) -> None:
    case = read_only_case
    revision = case.tool.approval_revision()
    call_id = await add_call(case, revision=revision, approval=None)
    case.tool.input_schema["required"] = ["changed"]
    case.supervisor.llm = CompletionLLM()
    await case.supervisor._run(case.session_id)
    assert case.client.calls == 0
    async with session_factory() as db:
        call = await db.get(ToolCall, call_id)
        assert call is not None and call.status == "error"
        assert call.result is not None and call.result["success"] is False


def test_read_only_mcp_tool_does_not_require_confirmation(
    read_only_case: ReadOnlyCase,
) -> None:
    case = read_only_case
    registry = ToolRegistry(tools=(case.tool,))
    assert registry.requires_confirmation(case.tool.name, "dev") is False

"""Integration tests for source-backed execution guards on real isolated SQLite."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update

from agent.database import init_database, session_factory
from agent.models import Project, Session, User, Workspace, WorkspaceMembership
from agent.source_execution import capture_source_revision, validate_source_execution
from agent.tool_sources import ToolSource
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.mcp_tools import McpBackedTool, McpToolPolicy


class CountingFakeClient:
    """A fake MCP client that never starts processes or models and counts calls."""

    def __init__(self) -> None:
        self.list_calls = 0
        self.call_calls = 0

    async def list_tools(self) -> list[Any]:
        self.list_calls += 1
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_calls += 1
        return {"content": [], "isError": False}


class BuiltinEcho(Tool):
    name = "builtin_echo"
    description = "Builtin echo"
    risk_level = RiskLevel.read_only
    input_schema = {"type": "object"}

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"content": []}


class _RefusingSession:
    async def execute(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("No SQL expected")


@dataclass(frozen=True)
class ExecCase:
    session: Session
    user: User
    membership: WorkspaceMembership
    workspace_id: str
    project_id: str


@pytest.fixture
async def exec_case(tmp_path: Path) -> ExecCase:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="Exec workspace")
        user = User(display_name="Exec user", enabled=True)
        db.add_all([workspace, user])
        await db.flush()
        membership = WorkspaceMembership(workspace_id=workspace.id, user_id=user.id, role="member")
        project = Project(name="Exec project", root_path=str(tmp_path), workspace_id=workspace.id)
        db.add_all([membership, project])
        await db.flush()
        session = Session(
            project_id=project.id,
            created_by_user_id=user.id,
            mode="dev",
            title="Exec session",
            configuration={},
        )
        db.add(session)
        await db.commit()
        return ExecCase(session, user, membership, workspace.id, project.id)


def _source(exec_case: ExecCase, **overrides: object) -> ToolSource:
    defaults: dict[str, object] = {
        "source_id": "server-1",
        "workspace_id": exec_case.workspace_id,
        "project_ids": [exec_case.project_id],
        "command": ("server",),
        "enabled": True,
    }
    return ToolSource.model_validate(defaults | overrides)


def _policy_for(source: ToolSource) -> McpToolPolicy:
    return McpToolPolicy(
        server_id=source.source_id,
        risk_level=source.risk_level,
        allowed_modes=source.allowed_modes,
        max_result_chars=source.max_result_chars,
    )


def _mcp_tool(
    client: CountingFakeClient,
    source: ToolSource,
    *,
    remote: str = "echo",
    input_schema: dict[str, Any] | None = None,
) -> McpBackedTool:
    return McpBackedTool(
        client=client,
        policy=_policy_for(source),
        name=remote,
        description="Echo",
        input_schema=input_schema or {"type": "object"},
        source=source,
    )


def test_capture_revision_builtin_is_none() -> None:
    assert capture_source_revision(BuiltinEcho()) is None


async def test_allowed_actor_passes(exec_case: ExecCase) -> None:
    client = CountingFakeClient()
    tool = _mcp_tool(client, _source(exec_case))
    approved = tool.approval_revision()
    async with session_factory() as db:
        await validate_source_execution(db, exec_case.session.id, tool, approved)
    assert client.list_calls == 0
    assert client.call_calls == 0


async def test_membership_revoked_by_other_txn_denies(exec_case: ExecCase) -> None:
    client = CountingFakeClient()
    tool = _mcp_tool(client, _source(exec_case))
    approved = tool.approval_revision()
    async with session_factory() as db:
        membership = await db.get(WorkspaceMembership, (exec_case.workspace_id, exec_case.user.id))
        user = await db.get(User, exec_case.user.id)
        assert membership is not None and user is not None
        await validate_source_execution(db, exec_case.session.id, tool, approved)
        async with session_factory() as other:
            await other.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == exec_case.workspace_id,
                    WorkspaceMembership.user_id == exec_case.user.id,
                )
                .values(revoked_at=datetime.now(UTC))
            )
            await other.commit()
        assert membership.revoked_at is None
        assert user.enabled is True
        with pytest.raises(ToolError):
            await validate_source_execution(db, exec_case.session.id, tool, approved)


@pytest.mark.parametrize("change", ["disabled", "mode-ask", "null-creator", "revoked"])
async def test_denied_facts_raise(exec_case: ExecCase, change: str) -> None:
    client = CountingFakeClient()
    tool = _mcp_tool(client, _source(exec_case))
    approved = tool.approval_revision()
    async with session_factory() as db:
        membership = await db.get(WorkspaceMembership, (exec_case.workspace_id, exec_case.user.id))
        session = await db.get(Session, exec_case.session.id)
        assert membership is not None and session is not None
        if change == "disabled":
            await db.execute(update(User).where(User.id == exec_case.user.id).values(enabled=False))
        elif change == "mode-ask":
            session.mode = "ask"
        elif change == "null-creator":
            session.created_by_user_id = None
        elif change == "revoked":
            membership.revoked_at = datetime.now(UTC)
        await db.commit()
        with pytest.raises(ToolError):
            await validate_source_execution(db, exec_case.session.id, tool, approved)


async def test_wrong_project_scope_denied(exec_case: ExecCase) -> None:
    client = CountingFakeClient()
    source = _source(exec_case, project_ids=["00000000-0000-0000-0000-000000000000"])
    tool = _mcp_tool(client, source)
    approved = tool.approval_revision()
    async with session_factory() as db:
        with pytest.raises(ToolError):
            await validate_source_execution(db, exec_case.session.id, tool, approved)


async def test_unbound_mcp_denied_before_sql() -> None:
    client = CountingFakeClient()
    policy = McpToolPolicy(server_id="server-1")
    tool = McpBackedTool(
        client=client,
        policy=policy,
        name="echo",
        description="Echo",
        input_schema={"type": "object"},
        source=None,
    )
    with pytest.raises(ToolError):
        await validate_source_execution(_RefusingSession(), "any-session", tool, "some-revision")


async def test_changed_schema_denied_before_sql(exec_case: ExecCase) -> None:
    client = CountingFakeClient()
    tool = _mcp_tool(client, _source(exec_case))
    approved = tool.approval_revision()
    tool.input_schema = {
        "type": "object",
        "properties": {"added": {"type": "string"}},
    }
    with pytest.raises(ToolError):
        await validate_source_execution(_RefusingSession(), "any-session", tool, approved)


async def test_changed_source_config_denied_before_sql(exec_case: ExecCase) -> None:
    client = CountingFakeClient()
    tool = _mcp_tool(client, _source(exec_case))
    approved = tool.approval_revision()
    changed = _mcp_tool(client, _source(exec_case, allowed_modes=frozenset({"ask"})))
    with pytest.raises(ToolError):
        await validate_source_execution(_RefusingSession(), "any-session", changed, approved)


async def test_missing_revision_denied_before_sql(exec_case: ExecCase) -> None:
    client = CountingFakeClient()
    tool = _mcp_tool(client, _source(exec_case))
    with pytest.raises(ToolError):
        await validate_source_execution(_RefusingSession(), "any-session", tool, None)


async def test_builtin_rejects_nonnull_revision_fail_closed() -> None:
    tool = BuiltinEcho()
    with pytest.raises(ToolError):
        await validate_source_execution(
            _RefusingSession(), "any-session", tool, "some-stored-revision"
        )


async def test_builtin_with_null_revision_passes_no_sql() -> None:
    await validate_source_execution(_RefusingSession(), "any-session", BuiltinEcho(), None)


@pytest.mark.parametrize("revision", [None, "stored-revision"])
async def test_builtin_cannot_impersonate_reserved_mcp_namespace(revision: str | None) -> None:
    tool = BuiltinEcho()
    tool.name = "mcp__source__echo"
    with pytest.raises(ToolError, match="namespace requires a trusted source"):
        capture_source_revision(tool)
    with pytest.raises(ToolError, match="namespace requires a trusted source"):
        await validate_source_execution(_RefusingSession(), "any-session", tool, revision)

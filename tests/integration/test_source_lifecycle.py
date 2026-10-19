"""Exercise per-run discovery with fresh SQL grants and isolated fake transports."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete

from agent import source_lifecycle
from agent.config import Settings
from agent.database import init_database, session_factory
from agent.mcp_client import McpTool, StdioMcpClient
from agent.models import Project, Session, User, Workspace, WorkspaceMembership
from agent.runtime import AgentSupervisor
from agent.sandbox import available
from agent.session_fence import CURRENT_FENCE, FenceLostError, claim
from agent.source_lifecycle import source_registry
from agent.tools.base import ToolError
from agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class LifecycleCase:
    session_id: str
    user_id: str
    workspace_id: str
    project_id: str
    root: Path
    settings: Settings


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, command: list[str], **options: Any) -> None:
        self.command = command
        self.options = options
        self.closed = False
        self.on_list: Callable[[], Awaitable[None]] | None = None
        self.instances.append(self)

    async def start(self) -> None:
        pass

    async def list_tools(self) -> list[McpTool]:
        if self.on_list is not None:
            await self.on_list()
        return [McpTool("lookup", "Lookup", {"type": "object"})]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"content": []}

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
async def lifecycle_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LifecycleCase:
    await init_database(recover_running=False)
    async with session_factory() as db:
        workspace = Workspace(name="Lifecycle workspace")
        user = User(display_name="Lifecycle user", enabled=True)
        db.add_all([workspace, user])
        await db.flush()
        db.add(WorkspaceMembership(workspace_id=workspace.id, user_id=user.id, role="member"))
        project = Project(
            name="Lifecycle project", root_path=str(tmp_path), workspace_id=workspace.id
        )
        db.add(project)
        await db.flush()
        session = Session(
            project_id=project.id, created_by_user_id=user.id, mode="dev", title="Lifecycle session"
        )
        db.add(session)
        await db.commit()
        path = tmp_path / "operator-sources.json"
        path.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "source_id": "catalog",
                            "workspace_id": workspace.id,
                            "project_ids": [project.id],
                            "enabled": True,
                            "command": ["operator-server"],
                            "env_keys": ["MCP_TEST_TOKEN"],
                        }
                    ]
                }
            )
        )
        settings = Settings(tool_sources_file=path, data_dir=tmp_path / "runtime")
        case = LifecycleCase(session.id, user.id, workspace.id, project.id, tmp_path, settings)
    FakeClient.instances = []
    monkeypatch.setattr(source_lifecycle, "StdioMcpClient", FakeClient)
    return case


async def test_allowed_source_is_bound_scoped_and_closed(
    lifecycle_case: LifecycleCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = lifecycle_case
    monkeypatch.setenv("MCP_TEST_TOKEN", "explicit-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-inherit")
    base = ToolRegistry(tools=())
    async with source_registry(case.settings, case.session_id, case.root, base) as registry:
        client = FakeClient.instances[0]
        assert client.options["sandbox"].mode == "required"
        assert client.options["sandbox"].allow_network is False
        assert client.options["cwd"] == case.root
        assert client.options["session_id"] == case.session_id
        assert client.options["env"]["MCP_TEST_TOKEN"] == "explicit-secret"
        assert set(client.options["env"]) <= {"PATH", "LANG", "MCP_TEST_TOKEN"}
        tool = registry.get("mcp__catalog__lookup", "dev")
        assert tool.source.workspace_id == case.workspace_id
        assert len(tool.approval_revision()) == 64
        assert not client.closed
        assert base.definitions("dev") == []
    assert client.closed


async def test_unconfigured_sources_do_not_construct_clients(lifecycle_case: LifecycleCase) -> None:
    case = lifecycle_case
    async with source_registry(Settings(), case.session_id, case.root, ToolRegistry(tools=())):
        assert FakeClient.instances == []


@pytest.mark.parametrize(
    "denial",
    [
        "creator",
        "workspace",
        "project",
        "mode",
        "user",
        "membership",
        "stop",
        "terminal",
        "root",
        "disabled",
    ],
)
async def test_denied_source_never_starts(lifecycle_case: LifecycleCase, denial: str) -> None:
    case = lifecycle_case
    async with session_factory() as db:
        session = await db.get(Session, case.session_id)
        project = await db.get(Project, case.project_id)
        user = await db.get(User, case.user_id)
        assert session is not None and project is not None and user is not None
        if denial == "creator":
            session.created_by_user_id = None
        elif denial == "workspace":
            project.workspace_id = None
        elif denial in {"project", "disabled"}:
            document = json.loads(case.settings.tool_sources_file.read_text())
            if denial == "project":
                document["sources"][0]["project_ids"] = ["another-project"]
            else:
                document["sources"][0]["enabled"] = False
            case.settings.tool_sources_file.write_text(json.dumps(document))
        elif denial == "mode":
            session.mode = "ask"
        elif denial == "user":
            user.enabled = False
        elif denial == "membership":
            await db.execute(
                delete(WorkspaceMembership).where(WorkspaceMembership.user_id == case.user_id)
            )
        elif denial == "stop":
            session.stop_requested = True
        elif denial == "terminal":
            session.status = "completed"
        elif denial == "root":
            project.root_path = str(case.root / "runtime")
            Path(project.root_path).mkdir()
        await db.commit()
    async with source_registry(
        case.settings, case.session_id, case.root, ToolRegistry(tools=())
    ) as registry:
        assert registry.definitions("dev") == []
        assert FakeClient.instances == []


@pytest.mark.parametrize("change", ["stop", "membership"])
async def test_access_changed_during_discovery_fails_closed(
    lifecycle_case: LifecycleCase, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    case = lifecycle_case

    async def revoke() -> None:
        async with session_factory() as db:
            if change == "stop":
                session = await db.get(Session, case.session_id)
                assert session is not None
                session.stop_requested = True
            else:
                user = await db.get(User, case.user_id)
                assert user is not None
                user.enabled = False
            await db.commit()

    def client_factory(*args: Any, **kwargs: Any) -> FakeClient:
        client = FakeClient(*args, **kwargs)
        client.on_list = revoke
        return client

    monkeypatch.setattr(source_lifecycle, "StdioMcpClient", client_factory)
    with pytest.raises(ToolError, match="could not be initialized"):
        async with source_registry(
            case.settings, case.session_id, case.root, ToolRegistry(tools=())
        ):
            pytest.fail("Registry must not be exposed after revocation")
    assert FakeClient.instances[0].closed


async def test_lost_fence_is_not_downgraded_to_initialization_error(
    lifecycle_case: LifecycleCase,
) -> None:
    case = lifecycle_case
    async with session_factory() as db:
        fence = await claim(db, case.session_id, lease_seconds=60)
        assert fence is not None
        session = await db.get(Session, case.session_id)
        assert session is not None
        session.owner_token = "new-owner"
        await db.commit()
    token = CURRENT_FENCE.set(fence)
    try:
        with pytest.raises(FenceLostError):
            async with source_registry(
                case.settings, case.session_id, case.root, ToolRegistry(tools=())
            ):
                pytest.fail("A stale owner must not discover tools")
        assert FakeClient.instances == []
    finally:
        CURRENT_FENCE.reset(token)


async def test_runtime_clients_are_per_run_and_registry_is_reset(
    lifecycle_case: LifecycleCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = lifecycle_case
    supervisor = AgentSupervisor(case.settings, ToolRegistry(tools=()))
    clients: list[Any] = []
    both_entered = asyncio.Event()

    async def run_session(session_id: str) -> None:
        client = supervisor._tools.get("mcp__catalog__lookup", "dev")._client
        clients.append(client)
        if len(clients) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=5)
        assert supervisor._tools.get("mcp__catalog__lookup", "dev")._client is client

    monkeypatch.setattr(supervisor, "_run_session", run_session)
    await asyncio.gather(supervisor._run(case.session_id), supervisor._run(case.session_id))
    assert len(clients) == 2 and clients[0] is not clients[1]
    assert all(client.closed for client in clients)
    assert supervisor._tools is supervisor.registry
    assert supervisor.registry.definitions("dev") == []


_SERVER = """
import json
import os
import sys
from pathlib import Path
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    result = {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {}}
    elif method == "tools/list":
        result = {"tools": [{"name": "read", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        try:
            text = Path(request["params"]["arguments"]["path"]).read_text()
        except PermissionError:
            text = "denied"
        result = {"content": [{"type": "text", "text": text}]}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


@pytest.mark.skipif(not available(), reason="no usable sandbox backend")
async def test_lifecycle_real_required_sandbox_and_cleanup(
    lifecycle_case: LifecycleCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = lifecycle_case
    monkeypatch.setattr(source_lifecycle, "StdioMcpClient", StdioMcpClient)
    path = case.settings.tool_sources_file
    assert path is not None
    document = json.loads(path.read_text())
    document["sources"][0]["command"] = [str(Path(sys.executable).resolve()), "-u", "-c", _SERVER]
    path.write_text(json.dumps(document))
    settings = case.settings.model_copy(
        update={"data_dir": case.root.parent / f"{case.root.name}-runtime"}
    )
    outside = case.root.parent / f"{case.root.name}-private.txt"
    outside.write_text("private")
    inside = case.root / "public.txt"
    inside.write_text("public")
    async with source_registry(
        settings, case.session_id, case.root, ToolRegistry(tools=())
    ) as registry:
        client = registry.get("mcp__catalog__read", "dev")._client
        process = client._process
        assert process is not None
        allowed = await client.call_tool("read", {"path": str(inside)})
        denied = await client.call_tool("read", {"path": str(outside)})
        assert allowed["content"][0]["text"] == "public"
        assert denied["content"][0]["text"] == "denied"
    assert outside.read_text() == "private"
    assert process.returncode is not None
    assert client._process is None

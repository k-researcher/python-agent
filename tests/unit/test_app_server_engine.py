"""Port contracts derived from the accepted protocol and generated schema."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from jsonschema import Draft7Validator, validate

from agent.config import EngineProfileSettings
from agent.engine import (
    ApprovalEvent,
    DiffEvent,
    DoneEvent,
    ErrorEvent,
    LimitsEvent,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    UserInputEvent,
)
from agent.engine import EngineSpec as Spec

FIXTURES = Path(__file__).parents[1] / "fixtures" / "app_server"
SCHEMA = json.loads((FIXTURES / "schema.json").read_text())
MODULE = SCHEMA["adapter_module"]
SCHEMAS = SCHEMA["schemas"]
EVENTS = json.loads((FIXTURES / "events.json").read_text())
NOTIFICATIONS = dict(
    zip(
        [
            "item/agentMessage/delta",
            "item/reasoning/summaryTextDelta",
            "item/reasoning/textDelta",
            "item/started",
            "item/completed",
            "turn/diff/updated",
            "thread/tokenUsage/updated",
            "account/rateLimits/updated",
            "turn/started",
            "turn/completed",
        ],
        [
            "AgentMessageDeltaNotification",
            "ReasoningSummaryTextDeltaNotification",
            "ReasoningTextDeltaNotification",
            "ItemStartedNotification",
            "ItemCompletedNotification",
            "TurnDiffUpdatedNotification",
            "ThreadTokenUsageUpdatedNotification",
            "AccountRateLimitsUpdatedNotification",
            "TurnStartedNotification",
            "TurnCompletedNotification",
        ],
        strict=True,
    )
)


def packet(method, **params):
    return {"method": method, "params": {"threadId": "thread-1", "turnId": "turn-1", **params}}


def request(method, **params):
    return {"id": 101, **packet(method, itemId="item-1", startedAtMs=1, **params)}


def done(status="completed"):
    return {
        "method": "turn/completed",
        "params": {"threadId": "thread-1", "turn": {"id": "turn-1", "items": [], "status": status}},
    }


@pytest.fixture
def peer(tmp_path, monkeypatch):
    """Only the explicitly created fake executable may cross the shared guard."""
    scenario_file, log = tmp_path / "scenario.json", tmp_path / "wire.jsonl"
    binary = tmp_path / "engine-cli"
    binary.write_text(
        f"#!{sys.executable}\nimport runpy,sys\n"
        f"sys.argv = [{str(FIXTURES / 'fake_cli.py')!r}, {str(scenario_file)!r}, "
        f"{str(log)!r}, *sys.argv[1:]]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    for name in [
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HTTPS_PROXY",
        "PYTHONPATH",
        "CODEX_HOME",
        "ANTHROPIC_AUTH_TOKEN",
        "LD_PRELOAD",
    ]:
        monkeypatch.setenv(name, "must-not-leak")
    from agent.engine_process import GuardedProcess

    original = GuardedProcess.start
    processes = []

    async def start(command, **kwargs):
        assert Path(command[0]) == binary
        process = await original(command, **kwargs)
        processes.append(process)
        return process

    spawn = asyncio.create_subprocess_exec

    async def guarded_spawn(*command, **kwargs):
        assert Path(command[0]) == Path(sys.executable).resolve()
        assert Path(command[1]).name == "engine_guard.py"
        assert str(binary) in command
        assert set(kwargs["env"]) <= {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
        assert kwargs["env"].get("TMPDIR")
        return await spawn(*command, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", guarded_spawn)
    monkeypatch.setattr(GuardedProcess, "start", start)

    def no_network(*args, **kwargs):
        pytest.fail("network forbidden")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)

    @asynccontextmanager
    async def create(scenario=None, **profile_changes):
        scenario_file.write_text(json.dumps(scenario or {}))
        profile = EngineProfileSettings(
            **(
                {
                    "enabled": True,
                    "adapter_module": MODULE,
                    "binary": binary.name,
                    "model": "configured-model",
                    "effort": "medium",
                    "operator_user_id": "owner",
                    "timeouts": {"open": 2, "request": 1},
                }
                | profile_changes
            )
        )
        engine = importlib.import_module(MODULE).Engine(profile)
        try:
            yield (
                engine,
                Spec(
                    tmp_path.resolve(), "dev", "session instructions", {"session_id": "session-1"}
                ),
                log,
            )
        finally:
            await engine.close()
            assert all(p.returncode is not None for p in processes)
            if log.exists():
                for row in wire(log):
                    if "validate" in row:
                        Draft7Validator(SCHEMAS[row["validate"]]).validate(row["value"])

    return create


def wire(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_recorded_notifications_follow_generated_schema():
    for event in EVENTS:
        validate(event["params"], SCHEMAS[NOTIFICATIONS[event["method"]]])


async def test_handshake_environment_isolation_and_resume(peer):
    async with asyncio.timeout(5), peer() as (engine, spec, log):
        health = await engine.health()
        assert (health.status, health.supported, health.auth_kind, health.isolation_capable) == (
            "ready",
            True,
            SCHEMA["plan_auth_kind"],
            True,
        ), health
        assert not any(p.get("method", "").startswith(("thread/", "turn/")) for p in wire(log))
        thread = await engine.open(spec, "stored-thread")
        assert thread.native_id == "stored-thread"
        rows = wire(log)
        methods = [p["method"] for p in rows if "method" in p]
        assert methods == ["initialize", "initialized", "account/read", "config/read"] * 2 + [
            "thread/resume"
        ]
        resume = next(p["params"] for p in rows if p.get("method") == "thread/resume")
        assert resume["threadId"] == "stored-thread"
        assert resume["approvalPolicy"] == "on-request"
        assert resume["approvalsReviewer"] == "user"
        assert resume["developerInstructions"] == spec.instructions
        assert "sandbox" not in resume  # legacy mode would disable restricted permissions
        for row in rows:
            if "env" in row:
                assert set(row["env"]) <= {
                    "PATH",
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "TMPDIR",
                    "__CF_USER_TEXT_ENCODING",
                }
                assert all(v != "must-not-leak" for v in row["env"].values())
                assert row["argv"] == ["--version"] or "--strict-config" in row["argv"]


@pytest.mark.parametrize(
    "scenario,status",
    [
        ({"version": "0.163.9"}, "degraded"),
        ({"version": "0.164.1"}, "degraded"),
        ({"version": "0.164.0-beta"}, "degraded"),
        ({"version": "bad"}, "degraded"),
        ({"auth": {"type": "apiKey"}}, "unavailable"),
        ({"auth": None}, "unavailable"),
        ({"auth": {"type": "newKind"}}, "unavailable"),
        ({"handshake": "malformed"}, "degraded"),
        ({"isolation": "missing"}, "unavailable"),
        ({"isolation": "legacy"}, "unavailable"),
    ],
)
async def test_health_fail_closed_without_turn(peer, scenario, status):
    async with asyncio.timeout(5), peer(scenario) as (engine, spec, log):
        health = await engine.health()
        assert health.status == status, health
        assert health.diagnostic
        assert not any(p.get("method", "").startswith(("thread/", "turn/")) for p in wire(log))
        with pytest.raises(RuntimeError):
            await engine.open(spec)


async def test_missing_binary(peer):
    async with asyncio.timeout(5), peer(binary="/missing/engine") as (engine, _, _):
        assert (await engine.health()).status == "unavailable"


async def test_mapping_final_dedupe_and_single_completion(peer):
    transcript = [{"send": p} for p in EVENTS + [EVENTS[2], done()]]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "new message", "submission-1")]
        for cls, text in [(TextEvent, "Hello world"), (ReasoningEvent, "ThinkDetails")]:
            matching = [e for e in events if isinstance(e, cls)]
            assert "".join(e.text for e in matching if e.phase == "delta") == text
            # finals replace the corresponding item in the existing runner
            assert "".join(e.text for e in matching if e.phase == "final") == text
            assert len([e for e in matching if e.phase == "final"]) == 1
        calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert {e.item_id for e in calls} == {"cmd-1", "file-1"}
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert {e.related_item_id for e in results} == {"cmd-1", "file-1"}
        assert all(e.success for e in results)
        diffs = [e for e in events if isinstance(e, DiffEvent)]
        assert any(e.paths == ["a.txt"] and e.status == "proposed" for e in diffs)
        assert any(e.paths == ["a.txt"] and e.status == "applied" for e in diffs)
        usage = next(e for e in events if isinstance(e, UsageEvent))
        assert (usage.input_tokens, usage.output_tokens, usage.cache_tokens) == (15, 3, 10)
        assert usage.reported_cost is None and usage.estimated_cost
        limits = next(e.limits for e in events if isinstance(e, LimitsEvent))
        assert limits.windows[0].used_share == 0.25
        assert limits.windows[0].duration_seconds == 18000
        assert limits.windows[1].used_share == 1
        assert limits.status == "exhausted"
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]
        assert all(e.submission_id == "submission-1" for e in events)
        assert [e.seq for e in events] == sorted({e.seq for e in events})
        turn = next(p["params"] for p in wire(log) if p.get("method") == "turn/start")
        assert turn == {
            "threadId": "thread-1",
            "input": [{"type": "text", "text": "new message", "text_elements": []}],
            "effort": "medium",
        }


@pytest.mark.parametrize("kind", ["commandExecution", "fileChange", "permissions"])
@pytest.mark.parametrize("decision", ["allow", "deny", "stop"])
async def test_each_approval_exact_response(peer, kind, decision):
    params = {"command": "pwd", "additionalPermissions": {}} if kind == "commandExecution" else {}
    expected = {"decision": {"allow": "accept", "deny": "decline", "stop": "cancel"}[decision]}
    if kind == "permissions":
        params = {"cwd": "/project", "permissions": {"network": {"enabled": True}}}
        expected = (
            {"permissions": {}, "scope": "turn"} if decision == "allow" else {"permissions": {}}
        )
    req = request(f"item/{kind}/requestApproval", **params)
    validate(
        req["params"],
        SCHEMAS[
            {
                "commandExecution": "CommandExecutionRequestApprovalParams",
                "fileChange": "FileChangeRequestApprovalParams",
                "permissions": "PermissionsRequestApprovalParams",
            }[kind]
        ],
    )
    transcript = [{"send": req}, {"receive": {"id": 101, "result": expected}}, {"send": done()}]
    if kind == "fileChange":
        transcript.insert(
            0,
            {
                "send": packet(
                    "item/started",
                    startedAtMs=1,
                    item={
                        "id": "item-1",
                        "type": "fileChange",
                        "status": "inProgress",
                        "changes": [{"path": "a.txt", "kind": {"type": "add"}, "diff": "+x"}],
                    },
                )
            },
        )
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        approvals = []
        async for event in engine.turn(thread, "x", "submission-1"):
            if isinstance(event, ApprovalEvent):
                approvals.append(event)
                assert event.scope == ("turn" if kind == "permissions" else "request")
                if decision == "stop":
                    await engine.interrupt(event.turn_id)
                else:
                    await engine.answer(event.request_id, decision)
                    await engine.answer(event.request_id, decision)  # late duplicate is inert
        assert len(approvals) == 1
        assert "acceptForSession" not in log.read_text()
        answers = [p for p in wire(log) if p.get("id") == 101 and "result" in p]
        assert [p["result"] for p in answers] == [expected]


async def test_grant_root_always_declined(peer):
    req = request("item/fileChange/requestApproval", grantRoot="/outside")
    transcript = [
        {"send": req},
        {"receive": {"id": 101, "result": {"decision": "decline"}}},
        {"send": done()},
    ]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, _):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert not any(isinstance(e, ApprovalEvent) for e in events)


async def test_permissions_intersection_and_symlink(peer, tmp_path):
    (tmp_path / "escape").symlink_to("/private")
    permissions = {
        "fileSystem": {
            "read": [str(tmp_path / "a"), "/outside"],
            "write": [str(tmp_path / "b"), str(tmp_path / "escape/x"), str(tmp_path / ".env")],
        },
        "network": {"enabled": True},
    }
    expected = {
        "permissions": {
            "fileSystem": {"read": [str(tmp_path / "a")], "write": [str(tmp_path / "b")]}
        },
        "scope": "turn",
    }
    req = request("item/permissions/requestApproval", cwd=str(tmp_path), permissions=permissions)
    transcript = [{"send": req}, {"receive": {"id": 101, "result": expected}}, {"send": done()}]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, _):
        thread = await engine.open(spec)
        async for event in engine.turn(thread, "x", "s"):
            if isinstance(event, ApprovalEvent):
                assert event.parameters["permissions"] == expected["permissions"]
                await engine.answer(event.request_id, "allow")


@pytest.mark.parametrize(
    "kind,result",
    [
        ("item/tool/requestUserInput", {"answers": {}}),
        ("mcpServer/elicitation/request", {"action": "decline"}),
        ("future/permission/request", None),
    ],
)
async def test_unsupported_requests_refused(peer, kind, result):
    req = (
        request(kind, questions=[], isBlocking=True)
        if "UserInput" in kind
        else {
            "id": 101,
            "method": kind,
            "params": {"threadId": "thread-1", "serverName": "disabled"},
        }
    )
    expected = {"id": 101, "result": result} if result else {"id": 101, "error": {"code": -32601}}
    transcript = [{"send": req}, {"receive": expected}, {"send": done()}]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, UserInputEvent):
                await engine.answer_user_input(event.request_id)
        assert not any(isinstance(e, ApprovalEvent) for e in events)
        if result is None:
            assert any(isinstance(e, ErrorEvent) for e in events)
            assert (await engine.health()).status == "degraded"
        if "UserInput" in kind:
            assert any(p.get("method") == "turn/interrupt" for p in wire(log))


@pytest.mark.parametrize(
    "step",
    [
        {"raw": "{"},
        {"raw": "[]"},
        {"raw": "x" * (1024 * 1024 + 1)},
        {"send": packet("item/agentMessage/delta", itemId="bad", delta=9)},
        {"send": packet("item/agentMessage/delta", itemId="bad")},
        {"exit": 0},
        {"exit": 7},
    ],
)
async def test_invalid_payload_eof_crash_are_error(peer, step):
    async with asyncio.timeout(5), peer({"transcript": [step]}) as (engine, spec, _):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert any(isinstance(e, ErrorEvent) for e in events)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["error"]


async def test_unknown_notification_optional(peer):
    transcript = [{"send": packet("future/optional", opaque=True)}, {"send": done()}]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, _):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert not any(isinstance(e, ErrorEvent) for e in events)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]


@pytest.mark.parametrize("status,outcome", [("failed", "error"), ("interrupted", "interrupted")])
async def test_terminal_status(peer, status, outcome):
    async with (
        asyncio.timeout(5),
        peer({"transcript": [{"send": done(status)}]}) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert [e.status for e in events if isinstance(e, DoneEvent)] == [outcome]


async def test_interrupt_is_native_and_bounded(peer):
    transcript = [{"send": packet("item/agentMessage/delta", itemId="a", delta="partial")}]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript, "wait_interrupt": True}) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, TextEvent):
                await engine.interrupt(event.turn_id)
        assert events[-1].status == "interrupted"
        interrupt = next(p for p in wire(log) if p.get("method") == "turn/interrupt")
        assert interrupt["params"] == {"threadId": "thread-1", "turnId": "turn-1"}


async def test_limits_unknown_and_no_probe_turn(peer):
    async with asyncio.timeout(5), peer() as (engine, _, log):
        limits = await engine.read_limits()
        assert limits is not None and limits.status == "unknown"
        assert not limits.windows and limits.observed_at is not None
        methods = [p["method"] for p in wire(log) if "method" in p]
        assert methods == ["initialize", "initialized", "account/read", "config/read"] * 2 + [
            "account/rateLimits/read",
        ]


def test_disabled_profile_never_imports_adapter_or_starts(monkeypatch):
    from agent.engine_adapters import EngineRegistry

    def forbidden(*args, **kwargs):
        pytest.fail("disabled engine accessed")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(asyncio, "create_task", forbidden)
    profile = EngineProfileSettings(adapter_module=MODULE)
    registry = EngineRegistry({"off": profile})
    assert not registry.available("off")
    assert registry.get("off") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"approvalPolicy": "never"},
        {"approvalsReviewer": "guardian_subagent"},
        {"activePermissionProfile": None},
        {"sandbox": {"type": "dangerFullAccess"}},
    ],
)
async def test_thread_cannot_widen_isolation(peer, overrides):
    async with asyncio.timeout(5), peer({"thread_overrides": overrides}) as (engine, spec, log):
        with pytest.raises(RuntimeError):
            await engine.open(spec)
        assert not any(p.get("method") == "turn/start" for p in wire(log))


@pytest.mark.parametrize(
    "extra",
    [
        {"additionalPermissions": {"network": {"enabled": True}}},
        {"additionalPermissions": {"fileSystem": {"write": ["/outside"]}}},
        {"networkApprovalContext": {"host": "outside.example", "protocol": "https"}},
        {"cwd": "/outside"},
    ],
)
async def test_command_escalation_cannot_be_approved(peer, extra):
    req = request("item/commandExecution/requestApproval", **extra)
    transcript = [
        {"send": req},
        {"receive": {"id": 101, "result": {"decision": "decline"}}},
        {"send": done()},
    ]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, ApprovalEvent):
                await engine.answer(event.request_id, "allow")
        assert [p["result"] for p in wire(log) if p.get("id") == 101 and "result" in p] == [
            {"decision": "decline"}
        ]
        assert events[-1].status == "completed"


@pytest.mark.parametrize("mode", ["read", "dev"])
async def test_process_uses_private_temp_and_restricted_profile(peer, mode):
    async with asyncio.timeout(5), peer() as (engine, spec, log):
        spec = Spec(spec.project_root, mode, spec.instructions, spec.settings)
        await engine.open(spec)
        for row in wire(log):
            if "argv" not in row or row["argv"] == ["--version"]:
                continue
            args = row["argv"]
            import tomllib

            config = {}
            for i, arg in enumerate(args):
                if arg == "-c":
                    config.update(tomllib.loads(args[i + 1]))
            profile = config["permissions"][config["default_permissions"]]
            fs = profile["filesystem"]
            assert fs[":root"] == "deny" and fs[":minimal"] == "read"
            assert fs[":tmpdir"] == "write" and fs[":slash_tmp"] == "deny"
            assert fs[row["cwd"]] == ("write" if mode == "dev" else "read") or (
                mode == "read" and fs[":workspace_roots"]["."] == "read"
            )
            assert profile["network"]["enabled"] is False
            temporary = Path(row["env"]["TMPDIR"])
            assert temporary != Path(os.environ.get("TMPDIR", "/tmp"))
            # Health probes are already reaped, including their private directories.
            if temporary.exists():
                assert temporary.stat().st_mode & 0o777 == 0o700


async def test_stop_cancellation_is_distinct_from_reject(peer):
    req = request("item/commandExecution/requestApproval", command="pwd", additionalPermissions={})
    transcript = [
        {"send": req},
        {"receive": {"id": 101, "result": {"decision": "cancel"}}},
        {"send": done()},
    ]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        async for event in engine.turn(thread, "x", "s"):
            if isinstance(event, ApprovalEvent):
                # The optional neutral cancellation port is used by runner teardown.
                await engine.cancel_approval(event.request_id)
                await engine.answer(event.request_id, "allow")
        assert len([p for p in wire(log) if p.get("id") == 101 and "result" in p]) == 1


@pytest.mark.parametrize(
    "kind,extra",
    [
        ("commandExecution", {"command": "sh", "cwd": "/project"}),
        ("fileChange", {}),
    ],
)
async def test_unscoped_tool_approval_is_refused(peer, kind, extra):
    req = request(f"item/{kind}/requestApproval", **extra)
    transcript = [
        {"send": req},
        {"receive": {"id": 101, "result": {"decision": "decline"}}},
        {"send": done()},
    ]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, ApprovalEvent):
                await engine.answer(event.request_id, "allow")
        assert [p["result"] for p in wire(log) if p.get("id") == 101 and "result" in p] == [
            {"decision": "decline"}
        ]
        assert events[-1].status == "completed"


async def test_files_outside_policy_cannot_be_approved(peer):
    item = {
        "id": "item-1",
        "type": "fileChange",
        "changes": [{"path": "/outside/file", "kind": {"type": "add"}, "diff": "+x"}],
        "status": "inProgress",
    }
    transcript = [
        {"send": packet("item/started", item=item, startedAtMs=1)},
        {"send": request("item/fileChange/requestApproval")},
        {"receive": {"id": 101, "result": {"decision": "decline"}}},
        {"send": done()},
    ]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, ApprovalEvent):
                await engine.answer(event.request_id, "allow")
        assert [p["result"] for p in wire(log) if p.get("id") == 101 and "result" in p] == [
            {"decision": "decline"}
        ]
        assert events[-1].status == "completed"


async def test_task_cancellation_interrupts_before_guard_reap(peer):
    async with (
        asyncio.timeout(5),
        peer(
            {
                "transcript": [
                    {"send": packet("item/agentMessage/delta", itemId="a", delta="partial")}
                ],
                "wait_interrupt": True,
            }
        ) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        entered = asyncio.Event()

        async def consume():
            async for event in engine.turn(thread, "x", "s"):
                if isinstance(event, TextEvent):
                    entered.set()

        task = asyncio.create_task(consume())
        try:
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert any(p.get("method") == "turn/interrupt" for p in wire(log))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("path", [".", ".git/config", ".npmrc", "prod.env", "private.key"])
async def test_permissions_do_not_reopen_protected_paths(peer, tmp_path, path):
    req = request(
        "item/permissions/requestApproval",
        cwd=str(tmp_path),
        permissions={"fileSystem": {"read": [str(tmp_path / path)]}},
    )
    transcript = [
        {"send": req},
        {"receive": {"id": 101, "result": {"permissions": {}, "scope": "turn"}}},
        {"send": done()},
    ]
    async with asyncio.timeout(5), peer({"transcript": transcript}) as (engine, spec, log):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, ApprovalEvent):
                await engine.answer(event.request_id, "allow")
        assert [p["result"] for p in wire(log) if p.get("id") == 101 and "result" in p] == [
            {"permissions": {}, "scope": "turn"}
        ]
        assert events[-1].status == "completed"

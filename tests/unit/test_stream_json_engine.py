"""Specification contracts; synthetic transport readiness is explicitly test-only."""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from agent.config import EngineProfileSettings
from agent.engine import (
    ApprovalEvent,
    DoneEvent,
    EngineSpec,
    ErrorEvent,
    Health,
    LimitsEvent,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "stream_json"
PROTOCOL = json.loads((FIXTURES / "protocol.json").read_text())
MODULE = PROTOCOL["adapter_module"]
EVENTS = json.loads((FIXTURES / "events.json").read_text())


def wire(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def result(**changes):
    return {**PROTOCOL["result"], **changes}


def approval(name="Read", arguments=None, **changes):
    return {
        "type": "control_request",
        "request_id": "approval-1",
        "request": {
            "subtype": "can_use_tool",
            "tool_name": name,
            "tool_use_id": "tool-1",
            "input": arguments or {"file_path": "a.txt", "offset": 1},
            "permission_suggestions": [
                {"type": "setMode", "mode": "bypassPermissions", "destination": "userSettings"}
            ],
            **changes,
        },
    }


@pytest.fixture
def peer(tmp_path, monkeypatch, request):
    scenario_file, log = tmp_path / "scenario.json", tmp_path / "wire.jsonl"
    log.touch()
    # Keep the executable's import path short and private. The project/scenario
    # still use pytest's workspace; fixture files precede protocol deadlines.
    binary_dir = tempfile.TemporaryDirectory(prefix="engine-peer-", dir="/private/tmp")
    request.addfinalizer(binary_dir.cleanup)
    binary = Path(binary_dir.name) / "engine-cli"
    binary.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"sys.argv = [sys.argv[0], {str(scenario_file)!r}, "
        f"{str(log)!r}, {str(FIXTURES / 'protocol.json')!r}, *sys.argv[1:]]\n"
        + (FIXTURES / "fake_cli.py").read_text()
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary.parent))
    for name in PROTOCOL["secret_environment"]:
        monkeypatch.setenv(name, "must-not-leak")
    from agent.engine_process import GuardedProcess

    processes = []
    original, spawn = GuardedProcess.start, asyncio.create_subprocess_exec

    async def start(command, **kwargs):
        assert Path(command[0]) == binary
        process = await original(command, **kwargs)
        processes.append(process)
        # Start request deadlines after this Python peer is ready to read. The
        # real guard still owns the process, pipes, group and bounded teardown.
        assert await process.stdout.readline() == b'{"peer_ready":true}\n'
        return process

    async def guarded_spawn(*command, **kwargs):
        assert Path(command[0]) == Path(sys.executable).resolve()
        assert Path(command[1]).name == "engine_guard.py"
        assert str(binary) in command
        assert set(kwargs["env"]) <= {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
        temporary = Path(kwargs["env"]["TMPDIR"])
        assert temporary.stat().st_mode & 0o777 == 0o700
        return await spawn(*command, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("network forbidden")

    monkeypatch.setattr(GuardedProcess, "start", start)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", guarded_spawn)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    @asynccontextmanager
    async def create(scenario=None, *, transport=False, **changes):
        scenario_file.write_text(json.dumps(scenario or {}))
        profile = EngineProfileSettings(
            **(
                {
                    "enabled": True,
                    "adapter_module": MODULE,
                    "binary": binary.name,
                    "model": "configured-model",
                    "operator_user_id": "owner",
                    "timeouts": {"open": 4, "request": 1},
                }
                | changes
            )
        )
        engine = importlib.import_module(MODULE).Engine(profile)
        if transport:
            # TEST ONLY readiness: ordinary transcripts cannot attest protection.
            # Health/version/auth are exercised separately, without a user turn.
            engine._health = Health(
                status="ready",
                binary=str(binary),
                version=PROTOCOL["version"],
                supported=True,
                auth_kind="subscription",
                isolation_capable=True,
            )
            engine._verify_isolation = lambda response: None
        try:
            yield (
                engine,
                EngineSpec(
                    tmp_path.resolve(), "dev", "session instructions", {"session_id": "session-1"}
                ),
                log,
            )
        finally:
            await engine.close()
            assert all(p.returncode is not None for p in processes)
            assert all(p.stderr_tail == b"" for p in processes)

    return create


async def test_limits_unknown_is_inert(peer):
    async with asyncio.timeout(5), peer() as (engine, _, log):
        assert await engine.read_limits() is None
        assert not wire(log)


@pytest.mark.parametrize("version", ["2.1.310", "2.1.312", "2.1.314", "2.1.311-beta", "bad"])
async def test_unsupported_versions_never_auth_or_start(peer, version):
    async with asyncio.timeout(5), peer({"version": version}) as (engine, spec, log):
        health = await engine.health()
        assert health.status == "degraded" and not health.supported, health
        with pytest.raises(RuntimeError):
            await engine.open(spec)
        assert [r["argv"] for r in wire(log) if "argv" in r] == [["--version"]]


@pytest.mark.parametrize(
    "auth",
    [
        None,
        {},
        {"loggedIn": False},
        {"loggedIn": True, "authMethod": "api_key"},
        {"loggedIn": True, "authMethod": "oauth_token"},
        {**PROTOCOL["auth"], "subscriptionType": None},
        {**PROTOCOL["auth"], "apiProvider": "unknown"},
    ],
)
async def test_auth_fail_closed(peer, auth):
    async with asyncio.timeout(5), peer({"auth": auth}) as (engine, spec, log):
        health = await engine.health()
        assert health.status == "unavailable" and not health.isolation_capable
        with pytest.raises(RuntimeError):
            await engine.open(spec)
        assert not any(r.get("type") == "user" for r in wire(log))
        assert not any("-p" in r.get("argv", []) for r in wire(log))


async def test_no_isolation_attestation_means_no_turn(peer):
    async with asyncio.timeout(5), peer() as (engine, spec, log):
        health = await engine.health()
        assert health.status == "unavailable" and health.supported, health
        assert not health.isolation_capable and "isolation" in health.diagnostic.lower()
        with pytest.raises(RuntimeError):
            await engine.open(spec)
        starts = [r["argv"] for r in wire(log) if "argv" in r]
        assert starts[:2] == [["--version"], ["auth", "status", "--json"]]
        assert starts[-1][0] == "-p"
        assert (await engine.health()).auth_kind == "subscription"
        assert not any(r.get("type") == "user" for r in wire(log))
        assert not any("effective" in str(r).lower() for r in wire(log))


async def test_missing_binary(peer):
    async with asyncio.timeout(5), peer(binary="/missing/engine") as (engine, _, _):
        assert (await engine.health()).status == "unavailable"


async def test_command_initialize_order_resume_and_clean_environment(peer):
    async with asyncio.timeout(5), peer(transport=True) as (engine, spec, log):
        thread = await engine.open(spec, "stored-session")
        assert thread.native_id == "stored-session"
        rows = wire(log)
        assert not any(r.get("type") == "user" for r in rows)
        starts = [r for r in rows if "argv" in r]
        assert len(starts) == 1
        assert starts[-1]["argv"][-2:] == ["--resume", "stored-session"]
        assert starts[-1]["cwd"] == str(spec.project_root)
        assert all("must-not-leak" not in str(r["env"]) for r in starts)
        requests = [r for r in rows if r.get("type") == "control_request"]
        assert all(r["request"] == {"subtype": "initialize", "hooks": None} for r in requests)
        assert "--include-partial-messages" not in starts[-1]["argv"]


@pytest.mark.parametrize("mode", ["read", "dev"])
async def test_bash_and_file_tool_policy_separately(peer, mode):
    async with asyncio.timeout(5), peer(transport=True) as (engine, spec, log):
        spec = replace(spec, mode=mode)
        await engine.open(spec)
        settings = [r["settings"] for r in wire(log) if "settings" in r][-1]
        sandbox = settings["sandbox"]
        assert sandbox["enabled"] is True
        assert sandbox["allowUnsandboxedCommands"] is False
        assert sandbox["autoAllowBashIfSandboxed"] is True
        fs = sandbox["filesystem"]
        assert fs["allowWrite"] == ([str(spec.project_root)] if mode == "dev" else [])
        assert str(spec.project_root) in fs["allowRead"]
        assert "/" in fs["denyRead"]
        assert any(".aws" in p for p in fs["denyRead"])
        assert sandbox["network"]["allowedDomains"] == []
        assert sandbox["network"]["allowLocalBinding"] is False
        permissions = settings["permissions"]
        assert permissions["defaultMode"] == "default"
        assert permissions["allow"] == []
        assert any(r.startswith("Read(") for r in permissions["deny"])
        assert any(".env" in r for r in permissions["deny"])
        assert "WebFetch" in permissions["deny"] and "Agent" in permissions["deny"]
        if mode == "read":
            assert "Write" in permissions["deny"] and "Edit" in permissions["deny"]
            assert str(spec.project_root) in fs["denyWrite"]
        assert settings["disableAllHooks"] is True and settings["hooks"] == {}
        assert settings["enabledPlugins"] == {}


@pytest.mark.parametrize(
    "changes",
    [
        {"apiKeySource": "env"},
        {"apiKeySource": None},
        {"tools": ["mcp__untrusted__run"]},
        {"tools": ["UnknownTool"]},
        {"mcp_servers": [{"name": "untrusted"}]},
        {"plugins": [{"name": "untrusted"}]},
        {"permissionMode": "bypassPermissions"},
        {"cwd": "/foreign"},
        {"model": "foreign-model"},
        {"tools": []},
        {"skills": ["untrusted"]},
        {"slash_commands": ["untrusted"]},
        {"session_id": ""},
    ],
)
async def test_init_refuses_unconfirmed_or_foreign_capabilities(peer, changes):
    async with asyncio.timeout(5), peer({"init": changes}, transport=True) as (engine, spec, log):
        with pytest.raises((RuntimeError, PermissionError)):
            await engine.open(spec)
        assert (await engine.health()).status in ("unavailable", "degraded")
        assert not any(r.get("type") == "user" for r in wire(log))


async def test_mapping_dedupe_tool_result_cache_limits_and_done(peer):
    transcript = [{"send": p} for p in [*EVENTS, EVENTS[5], result(), result()]]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "new message", "submission-1")]
        for kind, text in [(TextEvent, "Hello"), (ReasoningEvent, "Check")]:
            matching = [e for e in events if isinstance(e, kind)]
            assert [e.text for e in matching if e.phase == "delta"] == [text]
            assert [e.text for e in matching if e.phase == "final"] == [text]
            assert len({e.item_id for e in matching}) == 1
        assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 1
        tool = next(e for e in events if isinstance(e, ToolResultEvent))
        assert tool.related_item_id == "tool-1" and tool.output == "contents" and tool.success
        usage = next(e for e in events if isinstance(e, UsageEvent))
        assert (usage.input_tokens, usage.output_tokens, usage.cache_tokens) == (12, 3, 24)
        assert str(usage.reported_cost) == "0.012" and usage.estimated_cost
        limits = next(e.limits for e in events if isinstance(e, LimitsEvent))
        assert [w.used_share for w in limits.windows] == [0.25, 1]
        assert limits.status == "exhausted" and limits.observed_at
        assert await engine.read_limits() == limits
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]
        assert all(
            e.submission_id == "submission-1" and e.thread_id == thread.native_id for e in events
        )
        assert [e.seq for e in events] == sorted({e.seq for e in events})
        user = next(r for r in wire(log) if r.get("type") == "user")
        assert user["message"] == {"role": "user", "content": "session instructions\n\nnew message"}


@pytest.mark.parametrize("decision", ["allow", "deny", "stop", "ttl", "lease", "shutdown"])
async def test_approval_exact_input_and_deny_on_cancellation(peer, decision):
    req = approval()
    expected = (
        {"behavior": "allow", "updatedInput": req["request"]["input"]}
        if decision == "allow"
        else {"behavior": "deny", "message": "Permission denied"}
    )
    transcript = [{"send": req}, {"receive": expected}]
    if decision in ("stop", "lease", "shutdown"):
        transcript.append({"receive": "interrupt"})
    transcript.append({"send": result()})
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        async for event in engine.turn(thread, "x", "s"):
            if isinstance(event, ApprovalEvent):
                assert event.scope == "request" and event.item_id == "tool-1"
                assert "permission_suggestions" not in event.parameters
                if decision in ("stop", "lease"):
                    await engine.interrupt(event.turn_id)
                elif decision == "shutdown":
                    await engine.close()
                    break
                elif decision == "ttl":
                    await engine.cancel_approval(event.request_id)
                else:
                    # Mutating a port packet must not alter the native original.
                    event.parameters["input"]["offset"] = 100
                    await engine.answer(event.request_id, decision)
                await engine.answer(event.request_id, "allow")
        answers = [
            r["response"]["response"] for r in wire(log) if r.get("type") == "control_response"
        ]
        assert answers == [expected]


@pytest.mark.parametrize(
    "req",
    [
        approval("Write", {"file_path": "/outside/file", "content": "x"}),
        approval("Bash", {"command": "pwd", "dangerouslyDisableSandbox": True}),
        approval("Read", {"file_path": ".env"}),
        approval("WebFetch", {"url": "https://example.com"}),
        approval("Read", {"file_path": "a.txt"}, blocked_path="/outside"),
    ],
)
async def test_unsafe_requests_never_reach_approval(peer, req):
    transcript = [
        {"send": req},
        {"receive": {"behavior": "deny", "message": "Permission denied"}},
        {"send": result()},
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert not any(isinstance(e, ApprovalEvent) for e in events)


async def test_unknown_control_request_refused_and_degraded(peer):
    transcript = [
        {
            "send": {
                "type": "control_request",
                "request_id": "r",
                "request": {"subtype": "new_permission"},
            }
        },
        {"receive": {"behavior": "deny", "message": "Permission denied"}},
        {"receive": "interrupt"},
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert any(isinstance(e, ErrorEvent) for e in events)
        assert (await engine.health()).status == "degraded"
        assert any(r.get("response", {}).get("request_id") == "r" for r in wire(log))


@pytest.mark.parametrize(
    "step",
    [
        {"raw": "{"},
        {"raw": "[]"},
        {"raw": "x" * (1024 * 1024 + 1)},
        {"send": {"type": "assistant", "message": {}}},
        {"send": {"type": "result", "subtype": "new_result"}},
        {"exit": 0},
        {"exit": 7},
        {"send": {"type": "assistant", "session_id": "foreign", "message": {}}},
    ],
)
async def test_malformed_oversize_eof_crash_no_success(peer, step):
    async with (
        asyncio.timeout(5),
        peer({"transcript": [step]}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert any(isinstance(e, ErrorEvent) for e in events)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["error"]
        assert (await engine.health()).status == "degraded"


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"subtype": "success", "is_error": True}, "error"),
        ({"subtype": "error_max_turns", "is_error": True}, "error"),
        (
            {
                "subtype": "error_during_execution",
                "is_error": True,
                "terminal_reason": "aborted_streaming",
            },
            "interrupted",
        ),
    ],
)
async def test_result_status(peer, changes, status):
    async with (
        asyncio.timeout(5),
        peer({"transcript": [{"send": result(**changes)}]}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert [e.status for e in events if isinstance(e, DoneEvent)] == [status]


async def test_optional_unknown_notification_and_result_only_no_duplicate_text(peer):
    transcript = [{"send": {"type": "future_optional", "payload": {}}}, {"send": result()}]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert not any(isinstance(e, (TextEvent, ErrorEvent)) for e in events)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]


def test_profile_off_no_import_probe_or_process(monkeypatch):
    from agent.engine_adapters import EngineRegistry

    def forbidden(*args, **kwargs):
        pytest.fail("disabled profile must be inert")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(importlib.util, "find_spec", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    registry = EngineRegistry({"off": EngineProfileSettings(adapter_module=MODULE)})
    assert registry.get("off") is None
    assert not registry.available("off")


@pytest.mark.parametrize("auth", ["{", "[]", "null", '{"loggedIn": true}'])
async def test_auth_bad_or_unknown_packet_never_dispatches(peer, auth):
    async with asyncio.timeout(5), peer({"auth_raw": auth}) as (engine, spec, log):
        assert (await engine.health()).status in ("degraded", "unavailable")
        with pytest.raises(RuntimeError):
            await engine.open(spec)
        assert not any(r.get("type") == "user" for r in wire(log))


async def test_malformed_initialize_is_protocol_degraded(peer):
    async with asyncio.timeout(5), peer({"initialize": {}}) as (engine, _, _):
        assert (await engine.health()).status == "degraded"


async def test_interrupt_without_pending_approval(peer):
    transcript = [
        {"send": EVENTS[5]},
        {"receive": "interrupt"},
        {
            "send": result(
                subtype="error_during_execution", is_error=True, terminal_reason="aborted_streaming"
            )
        },
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        events = []
        interrupted = False
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if not interrupted:
                interrupted = True
                await engine.interrupt(event.turn_id)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["stopped"]
        assert any(r.get("request", {}).get("subtype") == "interrupt" for r in wire(log))


async def test_read_mode_and_symlink_requests_denied(peer, tmp_path):
    (tmp_path / "escape").symlink_to("/private")
    requests = [
        approval("Read", {"file_path": "escape/file"}),
        approval("Write", {"file_path": "a.txt", "content": "x"}),
    ]
    transcript = []
    for index, req in enumerate(requests):
        req["request_id"] = f"request-{index}"
        transcript += [
            {"send": req},
            {"receive": {"behavior": "deny", "message": "Permission denied"}},
        ]
    transcript.append({"send": result()})
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(replace(spec, mode="read"))
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert not any(isinstance(e, ApprovalEvent) for e in events)


@pytest.mark.parametrize(
    "packet",
    [
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed",
                "unifiedWindows": {"five_hour": {"utilization": 2}},
            },
        },
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed",
                "unifiedWindows": {"five_hour": {"utilization": True}},
            },
        },
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed",
                "unifiedWindows": {"five_hour": {"utilization": 0.5, "resetsAt": 10**100}},
            },
        },
        result(usage={"input_tokens": -1, "output_tokens": 0}),
        result(total_cost_usd=True),
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "late"},
            },
        },
        {
            "type": "control_response",
            "response": {"subtype": "success", "request_id": "foreign", "response": {}},
        },
    ],
)
async def test_required_payload_validation(peer, packet):
    async with (
        asyncio.timeout(5),
        peer({"transcript": [{"send": packet}]}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["error"]
        assert any(isinstance(e, ErrorEvent) for e in events)


async def test_late_stream_deltas_after_final_are_ignored(peer):
    late = json.loads(json.dumps(EVENTS[4]))
    late["event"]["delta"]["text"] = "duplicate"
    transcript = [{"send": p} for p in [*EVENTS, late, result()]]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert [e.text for e in events if isinstance(e, TextEvent) and e.phase == "delta"] == [
            "Hello"
        ]


async def test_last_limits_retained_after_bad_event_and_close(peer):
    observed = EVENTS[-1]
    bad = {"type": "rate_limit_event", "rate_limit_info": {"status": "future"}}
    transcript = [{"send": observed}, {"send": bad}]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        limits = next(e.limits for e in events if isinstance(e, LimitsEvent))
        await engine.close()
        assert await engine.read_limits() == limits
        assert (await engine.health()).status == "degraded"


async def test_denied_tool_result_links_to_permission_request_without_assistant_replay(peer):
    transcript = [
        {"send": approval()},
        {"receive": {"behavior": "deny", "message": "Permission denied"}},
        {
            "send": {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "denied",
                            "is_error": True,
                        }
                    ],
                },
            }
        },
        {"send": result()},
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if isinstance(event, ApprovalEvent):
                await engine.answer(event.request_id, "deny")
        assert len([e for e in events if isinstance(e, ToolCallEvent)]) == 1
        tool = next(e for e in events if isinstance(e, ToolResultEvent))
        assert tool.related_item_id == "tool-1" and not tool.success
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]


async def test_limits_without_unified_windows_preserve_reset_and_unknown_utilization(peer):
    packet = {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": "rejected",
            "rateLimitType": "five_hour",
            "resetsAt": 2000000000,
        },
    }
    async with (
        asyncio.timeout(5),
        peer({"transcript": [{"send": packet}, {"send": result()}]}, transport=True) as (
            engine,
            spec,
            _,
        ),
    ):
        thread = await engine.open(spec)
        events = [e async for e in engine.turn(thread, "x", "s")]
        limits = next(e.limits for e in events if isinstance(e, LimitsEvent))
        assert limits.status == "exhausted"
        assert len(limits.windows) == 1
        assert limits.windows[0].name == "five_hour"
        assert limits.windows[0].used_share is None
        assert limits.windows[0].reset_at.timestamp() == 2000000000


@pytest.mark.parametrize("cancellation", ["stop", "ttl"])
async def test_queued_approval_cannot_allow_after_cancellation(peer, cancellation):
    denied = {"behavior": "deny", "message": "Permission denied"}
    transcript = [{"send": approval()}, {"receive": denied}]
    if cancellation == "stop":
        transcript.append({"receive": "interrupt"})
    transcript.append({"send": result()})
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        tasks = []
        try:
            async for event in engine.turn(thread, "x", "s"):
                if not isinstance(event, ApprovalEvent):
                    continue
                queued, cancelled = asyncio.Event(), asyncio.Event()

                async def allow(ready=queued, request_id=event.request_id):
                    ready.set()
                    await engine.answer(request_id, "allow")

                async def cancel(
                    ready=cancelled, turn_id=event.turn_id, request_id=event.request_id
                ):
                    ready.set()
                    if cancellation == "stop":
                        await engine.interrupt(turn_id)
                    else:
                        await engine.cancel_approval(request_id)

                # Deterministic writer backpressure. Cancellation must be checked
                # when the decision is actually written, not before this wait.
                async with engine._write_lock:
                    tasks.append(asyncio.create_task(allow()))
                    await queued.wait()
                    tasks.append(asyncio.create_task(cancel()))
                    await cancelled.wait()
                await asyncio.gather(*tasks)
            answers = [
                r["response"]["response"] for r in wire(log) if r.get("type") == "control_response"
            ]
            assert answers == [denied]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("answer_at", ["usage", "done"])
async def test_terminal_result_revokes_pending_before_any_terminal_event(peer, answer_at):
    denied = {"behavior": "deny", "message": "Permission denied"}
    transcript = [
        {"send": approval()},
        {"send": result()},
        {"receive": denied},
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript, "exit_after_transcript": True}, transport=True) as (
            engine,
            spec,
            log,
        ),
    ):
        thread = await engine.open(spec)
        events = []
        async for event in engine.turn(thread, "x", "s"):
            events.append(event)
            if event.kind == answer_at:
                await engine.answer("approval-1", "allow")
        # A later answer cannot reopen a completed request either.
        await engine.answer("approval-1", "allow")
        await engine._process.wait()
        await engine.close()
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]
        assert [
            r["response"]["response"] for r in wire(log) if r.get("type") == "control_response"
        ] == [denied]


async def test_resumed_session_dispatch_preserves_native_identity(peer):
    native = "stored-session"
    async with (
        asyncio.timeout(5),
        peer({"transcript": [{"send": result(session_id=native)}]}, transport=True) as (
            engine,
            spec,
            log,
        ),
    ):
        thread = await engine.open(spec, native)
        events = [e async for e in engine.turn(thread, "continue", "submission-2")]
        assert all(e.thread_id == native and e.submission_id == "submission-2" for e in events)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["completed"]
        messages = [r["type"] for r in wire(log) if "type" in r]
        assert messages == ["control_request", "user"]


async def test_terminal_result_cancels_allow_queued_behind_writer(peer, monkeypatch):
    denied = {"behavior": "deny", "message": "Permission denied"}
    transcript = [
        {"send": approval()},
        {"send": result()},
        {"receive": denied},
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript, "exit_after_transcript": True}, transport=True) as (
            engine,
            spec,
            log,
        ),
    ):
        thread = await engine.open(spec)
        stream = engine.turn(thread, "x", "s")
        queued, parsed = asyncio.Event(), asyncio.Event()
        original = engine._result

        def parse(packet):
            events = original(packet)
            parsed.set()
            return events

        monkeypatch.setattr(engine, "_result", parse)

        async def allow():
            queued.set()
            await engine.answer("approval-1", "allow")

        tasks = []
        try:
            async for event in stream:
                if isinstance(event, ApprovalEvent):
                    break
            async with engine._write_lock:
                tasks.append(asyncio.create_task(allow()))
                await queued.wait()
                tasks.append(asyncio.create_task(anext(stream)))
                await parsed.wait()
            await asyncio.gather(*tasks)
            assert [e.status async for e in stream if isinstance(e, DoneEvent)] == ["completed"]
            await engine._process.wait()
            await engine.close()
            assert [
                r["response"]["response"] for r in wire(log) if r.get("type") == "control_response"
            ] == [denied]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await stream.aclose()


async def test_approval_rechecks_symlink_before_allow(peer, tmp_path):
    denied = {"behavior": "deny", "message": "Permission denied"}
    transcript = [
        {"send": approval()},
        {"receive": denied},
        {"send": result()},
    ]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, log),
    ):
        thread = await engine.open(spec)
        async for event in engine.turn(thread, "x", "s"):
            if isinstance(event, ApprovalEvent):
                (tmp_path / "a.txt").symlink_to("/private/outside.txt")
                await engine.answer(event.request_id, "allow")
        assert [
            r["response"]["response"] for r in wire(log) if r.get("type") == "control_response"
        ] == [denied]


async def test_broken_channel_during_terminal_deny_still_finishes_once(peer, monkeypatch):
    transcript = [{"send": approval()}, {"send": result()}]
    async with (
        asyncio.timeout(5),
        peer({"transcript": transcript}, transport=True) as (engine, spec, _),
    ):
        thread = await engine.open(spec)
        write = engine._process.stdin.write

        def broken_deny(raw):
            if json.loads(raw).get("type") == "control_response":
                raise BrokenPipeError("peer channel closed")
            write(raw)

        monkeypatch.setattr(engine._process.stdin, "write", broken_deny)
        events = [e async for e in engine.turn(thread, "x", "s")]
        assert any(isinstance(e, ErrorEvent) for e in events)
        assert [e.status for e in events if isinstance(e, DoneEvent)] == ["error"]

"""Local server lifecycle contracts without changing a running installation."""

from __future__ import annotations

import io
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import local_server as module
from scripts.local_server import Configuration, LocalServer, read_private, write_private


@pytest.fixture
def server(tmp_path):
    checkout, data = tmp_path / "checkout", tmp_path / "state"
    checkout.mkdir()
    data.mkdir()
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/pyagent").write_text("test executable")
    database = data / "agent.db"
    with sqlite3.connect(database) as db:
        db.executescript("""
            create table sessions(id text, status text);
            insert into sessions values ('s1', 'completed');
            create table projects(id text);
            insert into projects values ('p1');
            create table alembic_version(version_num text);
            insert into alembic_version values ('0025');
            create table mcp_servers(id text, url text, header_secrets text, http_protocol text);
            insert into mcp_servers values ('local', null, '{}', 'auto');
        """)
    return LocalServer(Configuration(checkout.resolve(), data.resolve(), database.resolve(), 18000))


def test_configuration_round_trip(server):
    server.config.validate()
    assert Configuration.from_payload(server.config.payload()) == server.config


@pytest.mark.parametrize("port", [0, 65536])
def test_invalid_port_is_refused(server, port):
    with pytest.raises(RuntimeError, match="port"):
        replace(server.config, port=port).validate()


def test_data_inside_checkout_is_refused(server):
    nested = server.config.checkout / "data"
    nested.mkdir()
    db = nested / "agent.db"
    db.touch()
    with pytest.raises(RuntimeError, match="separate"):
        replace(server.config, data_dir=nested, database=db).validate()


def test_database_outside_data_directory_is_refused(server):
    with pytest.raises(RuntimeError, match="inside"):
        replace(server.config, data_dir=server.config.checkout).validate()


def test_environment_fixes_source_loopback_database_and_required_mode(server, monkeypatch):
    for key in (
        "AGENT_HOST",
        "AGENT_PORT",
        "AGENT_SHELL_SANDBOX",
        "AGENT_DATABASE_URL",
        "PYTHONPATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ):
        monkeypatch.setenv(key, "wrong")
    env = server.environment()
    assert env["AGENT_HOST"] == "127.0.0.1"
    assert env["AGENT_PORT"] == "18000"
    assert env["AGENT_SHELL_SANDBOX"] == "required"
    assert env["AGENT_DATABASE_URL"] == f"sqlite+aiosqlite:///{server.config.database}"
    assert env["PYTHONPATH"] == str(server.config.checkout / "src")
    assert env["AGENT_ENV_FILE"] == env["AGENT_MODELS_FILE"] == ""
    assert "LD_PRELOAD" not in env and "DYLD_INSERT_LIBRARIES" not in env


def test_preflight_refuses_nested_sandbox(server, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=71))
    with pytest.raises(RuntimeError, match="child sandboxes"):
        server.preflight()


@pytest.mark.parametrize("operation", ["start", "update"])
def test_failed_preflight_has_no_launch_or_preparation(server, monkeypatch, operation):
    monkeypatch.setattr(server, "preflight", Mock(side_effect=RuntimeError("unavailable")))
    run, spawn = Mock(), Mock()
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="unavailable"):
        if operation == "start":
            server.start()
        else:
            server.update("restart", server.config.checkout, "same", "same")
    run.assert_not_called()
    spawn.assert_not_called()
    assert not (server.config.data_dir / "backups").exists()


def test_busy_port_does_not_spawn(server, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda: None)
    monkeypatch.setattr(server, "listeners", lambda: {99})
    spawn = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="occupied"):
        server.start()
    spawn.assert_not_called()


def test_listener_inspection_failure_is_not_an_empty_port(server, monkeypatch):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stderr="permission denied", stdout=""),
    )
    with pytest.raises(RuntimeError, match="inspect"):
        server.listeners()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file locks")
def test_parallel_update_is_refused(server):
    with server.lock(), pytest.raises(BlockingIOError), server.lock():
        pytest.fail("parallel controller acquired the lock")


@pytest.mark.skipif(os.name != "posix", reason="POSIX file locks")
def test_database_cannot_be_reused_with_another_port(server):
    with server.lock():
        pass
    another = LocalServer(replace(server.config, port=18001))
    with pytest.raises(RuntimeError, match="another checkout or port"), another.lock():
        pytest.fail("database identity changed")


@pytest.mark.parametrize("status", ["pending", "running", "awaiting_confirmation"])
def test_active_sessions_are_refused(server, status):
    with sqlite3.connect(server.config.database) as db:
        db.execute("update sessions set status=?", (status,))
    with pytest.raises(RuntimeError, match="Active sessions"):
        server.snapshot()


def test_snapshot_detects_setting_changes_without_a_count_change(server):
    before = server.snapshot()
    with sqlite3.connect(server.config.database) as db:
        db.execute("update mcp_servers set http_protocol='legacy'")
    after = server.snapshot()
    assert before["counts"] == after["counts"]
    assert before["fingerprints"]["mcp_servers"] != after["fingerprints"]["mcp_servers"]
    assert before["revision"] == "0025"


def test_private_receipt_refuses_symlink_and_wide_permissions(tmp_path):
    path = tmp_path / "receipt.json"
    write_private(path, {"pid": 7})
    assert read_private(path) == {"pid": 7}
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        read_private(link)
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="private"):
        read_private(path)


def test_profile_escapes_paths_and_has_no_network_permission(server, monkeypatch, tmp_path):
    common = tmp_path / 'common"directory'
    monkeypatch.setattr(module, "git", lambda *a: str(common))
    profile = server.preparation_profile(tmp_path / "temporary")
    assert '(subpath "' + str(server.config.checkout) + '")' in profile
    assert '\\"directory' in profile
    assert "(deny default)" in profile and "network" not in profile
    with pytest.raises(RuntimeError, match="Invalid path"):
        module.quote_path(Path("/bad\npath"))
    monkeypatch.setattr(module, "git", lambda *a: "/")
    with pytest.raises(RuntimeError, match="too broad"):
        server.preparation_profile(tmp_path / "temporary")


@pytest.mark.parametrize("pids,owned", [(set(), True), ({1, 2}, True), ({1}, False)])
def test_invalid_listener_is_refused_before_backup(server, monkeypatch, pids, owned):
    monkeypatch.setattr(
        module, "git", lambda checkout, *args: "same" if args[0] == "rev-parse" else ""
    )
    monkeypatch.setattr(server, "listeners", lambda: pids)
    monkeypatch.setattr(server, "owns_listener", lambda _: owned)
    with pytest.raises(RuntimeError, match="listener|Listener"):
        server.validate_update("restart", server.config.checkout, "same", "same")


def test_changed_commit_is_refused(server, monkeypatch):
    monkeypatch.setattr(module, "git", lambda *a: "different")
    with pytest.raises(RuntimeError, match="base changed"):
        server.validate_update("restart", server.config.checkout, "same", "same")


def test_frontend_cannot_use_backend_deployment(server, monkeypatch):
    def git(checkout, *args):
        if args[0] == "status":
            return ""
        if args[0] == "diff":
            return "frontend/src/App.vue"
        return "old" if checkout == server.config.checkout and len(args) == 2 else "new"

    monkeypatch.setattr(module, "git", git)
    with pytest.raises(RuntimeError, match="Frontend"):
        server.validate_update("backend", server.config.data_dir, "old", "new")


def test_health_must_belong_to_the_new_process(server, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda: None)
    states = iter([set(), {999}])
    monkeypatch.setattr(server, "listeners", lambda: next(states))
    clock = iter([0, 0, 31])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    process = Mock(pid=111)
    process.poll.return_value = None
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    opener = Mock()
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *a: opener)
    with pytest.raises(RuntimeError, match="timed out"):
        server.start()
    opener.open.assert_not_called()
    process.send_signal.assert_called_once()
    process.wait.assert_called_once()


def test_owned_listener_with_health_accepts_start(server, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda: None)
    states = iter([set(), {111}])
    monkeypatch.setattr(server, "listeners", lambda: next(states))
    monkeypatch.setattr(server, "owns_listener", lambda pid: pid == 111)
    process = Mock(pid=111)
    process.poll.return_value = None
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    response = io.BytesIO(b'{"status":"ok"}')
    response.status = 200
    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda *a: SimpleNamespace(open=lambda *a, **kw: response),
    )
    assert server.start() == 111
    process.send_signal.assert_not_called()


def test_prepare_backs_up_and_never_starts_a_server(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "validate_update", lambda *a: 7)
    monkeypatch.setattr(module, "git", lambda *a: "")
    signals = []

    def kill(pid, sig):
        assert pid == 7
        signals.append(sig)
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(module.os, "kill", kill)
    spawn = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    backup = tmp_path / "backup"
    before = server.snapshot()
    server.prepare("restart", server.config.checkout, "same", "same", backup)
    receipt = read_private(backup / "prepared.json")
    assert receipt["state"] == before
    assert (backup / "database.sqlite").stat().st_mode & 0o777 == 0o600
    assert signals == [module.signal.SIGTERM, 0]
    spawn.assert_not_called()


@pytest.mark.parametrize("valid", [True, False])
def test_sandboxed_preparation_and_receipt_precede_host_start(server, monkeypatch, valid):
    events = []
    monkeypatch.setattr(server, "preflight", lambda: events.append("preflight"))
    monkeypatch.setattr(server, "validate_update", lambda *a: 7)
    monkeypatch.setattr(server, "listeners", set)
    state = server.snapshot()

    def git(checkout, *args):
        return str(checkout / ".git") if "--git-common-dir" in args else "same"

    monkeypatch.setattr(module, "git", git)

    def prepare(command, **kwargs):
        assert command[:2] == ["/usr/bin/sandbox-exec", "-f"]
        assert "network" not in Path(command[2]).read_text()
        payload = read_private(Path(command[-1]))
        backup = Path(payload["backup"])
        backup.mkdir()
        write_private(
            backup / "prepared.json",
            {
                "kind": "restart",
                "from": "same",
                "to": "same",
                "old_pid": 7 if valid else 8,
                "state": state,
            },
        )
        events.append("prepare")

    monkeypatch.setattr(module.subprocess, "run", prepare)
    monkeypatch.setattr(server, "start", lambda: events.append("start") or 9)
    if valid:
        result = server.update("restart", server.config.checkout, "same", "same")
        assert result["pid"] == 9
        assert events == ["preflight", "prepare", "start"]
    else:
        with pytest.raises(RuntimeError, match="receipt"):
            server.update("restart", server.config.checkout, "same", "same")
        assert events == ["preflight", "prepare"]

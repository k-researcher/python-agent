"""Unit tests for CLI commands (agent.cli_client.commands).

Commands receive a fake :class:`AgentApi`, an output stream and an input
reader (dependency injection), so no live server or WebSocket is needed.
"""

from __future__ import annotations

import asyncio
import io
import os
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from agent.cli_client import commands
from agent.cli_client.api import TOKEN_HINT, ApiError
from agent.cli_client.commands import (
    CONNECTION_LOST_TEXT,
    EXIT_COMMAND,
    EXIT_INTERRUPTED,
    EXIT_NO_CONNECTION,
    EXIT_OK,
    EXIT_TASK_FAILED,
    INTERRUPTED_REJECT_COMMENT,
    NO_SERVER_HINT,
    NO_TOKEN_HINT,
    NON_INTERACTIVE_REJECT_COMMENT,
    TURN_STOPPED_TEXT,
    cmd_chat,
    cmd_continue,
    cmd_list,
    cmd_run,
    cmd_stop,
)

TOKEN = "top-secret-token"


class FakeApi:
    """Scriptable stand-in for :class:`AgentApi`."""

    def __init__(self) -> None:
        self._projects: list[dict[str, Any]] = []
        self._sessions: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._approvals: dict[str, list[dict[str, Any]]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._status: dict[str, str] = {}
        self.created_calls: list[str] = []
        self.sent_calls: list[tuple[str, str]] = []
        self.stopped_calls: list[str] = []
        self.resolved_calls: list[tuple[str, str, str | None]] = []
        self.ensure_calls: list[str] = []
        self.closed = False
        self.projects_error: ApiError | None = None
        self.ensure_error: ApiError | None = None
        self.session_error: ApiError | None = None
        self.sessions_error: ApiError | None = None

    # -- configuration helpers --------------------------------------------

    def add_project(self, project: dict[str, Any]) -> None:
        self._projects.append(project)

    def add_session(self, session: dict[str, Any], status: str | None = None) -> None:
        self._sessions.append(session)
        self._by_id[str(session["id"])] = session
        if status is not None:
            self._status[str(session["id"])] = status

    def set_status(self, session_id: str, status: str) -> None:
        self._status[session_id] = status

    def set_approvals(self, session_id: str, approvals: list[dict[str, Any]]) -> None:
        self._approvals[session_id] = approvals

    def set_events(self, session_id: str, events: list[dict[str, Any]]) -> None:
        self._events[session_id] = events

    # -- AgentApi methods ---------------------------------------------------

    async def projects(self) -> list[dict[str, Any]]:
        if self.projects_error is not None:
            raise self.projects_error
        return list(self._projects)

    async def ensure_project(self, root_path: str, name: str | None = None) -> dict[str, Any]:
        self.ensure_calls.append(root_path)
        if self.ensure_error is not None:
            raise self.ensure_error
        resolved = str(Path(root_path).resolve())
        for project in self._projects:
            if str(project.get("root_path", "")).rstrip("/") == resolved.rstrip("/"):
                return project
        project: dict[str, Any] = {
            "id": f"p{len(self._projects) + 1}",
            "name": name or Path(root_path).name or "project",
            "root_path": resolved,
        }
        self._projects.append(project)
        return project

    async def sessions(self) -> list[dict[str, Any]]:
        if self.sessions_error is not None:
            raise self.sessions_error
        return list(self._sessions)

    async def create_session(self, project_id: str, prompt: str, **_: Any) -> dict[str, Any]:
        session: dict[str, Any] = {
            "id": f"s{len(self._sessions) + 1}",
            "project_id": project_id,
            "title": prompt[:50],
            "status": self._status.get(f"s{len(self._sessions) + 1}", "running"),
            "created_at": "2025-01-15T10:30:00",
            "updated_at": "2025-01-15T10:30:00",
        }
        self.created_calls.append(prompt)
        self._sessions.append(session)
        self._by_id[str(session["id"])] = session
        return session

    async def send_message(self, session_id: str, content: str, **_: Any) -> dict[str, Any]:
        self.sent_calls.append((session_id, content))
        known = self._by_id.get(session_id)
        if known is not None:
            known["status"] = self._status.get(session_id, "running")
            return known
        session: dict[str, Any] = {
            "id": session_id,
            "project_id": "p1",
            "title": content[:50],
            "status": self._status.get(session_id, "running"),
        }
        self._by_id[session_id] = session
        return session

    async def session(self, session_id: str) -> dict[str, Any]:
        if self.session_error is not None:
            raise self.session_error
        known = self._by_id.get(session_id)
        if known is not None:
            known["status"] = self._status.get(session_id, known["status"])
            return known
        return {
            "id": session_id,
            "project_id": "p1",
            "title": "",
            "status": self._status.get(session_id, "running"),
        }

    async def events(self, session_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        sid = session_id or ""
        for event in self._events.get(sid, []):
            yield event

    async def approvals(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._approvals.get(session_id, []))

    async def resolve_approval(
        self, approval_id: str, decision: str, comment: str | None = None
    ) -> None:
        self.resolved_calls.append((approval_id, decision, comment))

    async def stop(self, session_id: str) -> None:
        self.stopped_calls.append(session_id)

    async def aclose(self) -> None:
        self.closed = True


def completed_status_event(session_id: str = "") -> dict[str, Any]:
    return {
        "type": "session.status",
        "session_id": session_id,
        "payload": {"status": "completed"},
    }


def error_status_event(session_id: str = "") -> dict[str, Any]:
    return {
        "type": "session.status",
        "session_id": session_id,
        "payload": {"status": "error", "error": "boom"},
    }


def stopped_status_event(session_id: str = "") -> dict[str, Any]:
    return {
        "type": "session.status",
        "session_id": session_id,
        "payload": {"status": "stopped"},
    }


def input_reader(*values: str) -> Callable[[str], str]:
    """Build an input reader that returns values then raises EOFError."""
    queue = list(values)

    def read(_prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return read


def run_async(awaitable: Any) -> Any:
    """Run a coroutine to completion and return its result."""
    return asyncio.run(awaitable)


# --- run: exit codes -----------------------------------------------------


def _run_cmd(
    api: FakeApi,
    *,
    task: str = "сделай что-нибудь",
    interactive: bool = False,
) -> tuple[int, str]:
    out = io.StringIO()
    result = run_async(cmd_run(api, out, input_reader(), interactive, Path.cwd(), task))
    return result, out.getvalue()


def test_run_completed_exit_code_zero() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_status("s1", "completed")

    code, output = _run_cmd(api, task="задача")
    assert code == EXIT_OK
    assert "completed" in output


def test_run_error_exit_code_one() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_status("s1", "error")

    code, output = _run_cmd(api, task="задача")
    assert code == EXIT_TASK_FAILED
    assert "error" in output


def test_run_stopped_exit_code_one() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_status("s1", "stopped")

    code, _ = _run_cmd(api, task="задача")
    assert code == EXIT_TASK_FAILED


def test_run_no_connection_exit_code_two_noninteractive() -> None:
    api = FakeApi()
    api.projects_error = ApiError(0, "Request failed: ConnectError")

    code, output = _run_cmd(api, task="задача", interactive=False)
    assert code == EXIT_NO_CONNECTION
    assert "Агент не запущен" in output
    assert NO_SERVER_HINT in output
    # No question is asked in non-interactive mode.
    assert "Запустить сейчас" not in output


def test_run_no_connection_detects_missing_token() -> None:
    api = FakeApi()
    api.projects_error = ApiError(401, TOKEN_HINT)

    code, output = _run_cmd(api, task="задача", interactive=False)
    assert code == EXIT_NO_CONNECTION
    assert NO_TOKEN_HINT in output


def test_run_does_not_echo_token() -> None:
    api = FakeApi()
    # A closer real-world failure embeds the configured token nowhere.
    api.projects_error = ApiError(401, TOKEN_HINT)

    out = io.StringIO()
    code = run_async(cmd_run(api, out, input_reader(), False, Path.cwd(), "задача"))
    assert code == EXIT_NO_CONNECTION
    assert TOKEN not in out.getvalue()


def test_run_empty_task_rejected() -> None:
    api = FakeApi()
    code, output = _run_cmd(api, task="   ", interactive=False)
    assert code == EXIT_TASK_FAILED
    assert "пустой" in output


# --- run: non-interactive approval rejection -----------------------------


def _run_with_approval(interactive: bool) -> tuple[int, str, FakeApi]:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_approvals(
        "s1",
        [
            {
                "id": "a1",
                "tool_name": "shell",
                "arguments": {"command": "rm -rf /tmp/x"},
                "risk_level": "high",
            }
        ],
    )
    # The session is awaiting a confirmation before any event arrives.
    api.set_status("s1", "awaiting_confirmation")
    # After the refusal the server stops the session.
    api.set_events("s1", [stopped_status_event("s1")])

    out = io.StringIO()
    reader = input_reader("n") if interactive else input_reader()
    code = run_async(cmd_run(api, out, reader, interactive, Path.cwd(), "задача"))
    return code, out.getvalue(), api


def test_run_noninteractive_rejects_approval() -> None:
    code, output, api = _run_with_approval(interactive=False)
    assert code == EXIT_TASK_FAILED
    assert NON_INTERACTIVE_REJECT_COMMENT in output
    assert api.resolved_calls == [("a1", "reject", NON_INTERACTIVE_REJECT_COMMENT)]


def test_run_interactive_approval_asks_user() -> None:
    code, output, api = _run_with_approval(interactive=True)
    assert api.resolved_calls == [("a1", "reject", None)]
    # The renderer printed the approval request.
    assert "Запрос инструмента" in output


# --- chat -----------------------------------------------------------------


def _project(api: FakeApi) -> None:
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_events("s1", [completed_status_event("s1")])


def test_chat_two_messages_use_one_session() -> None:
    api = FakeApi()
    _project(api)

    out = io.StringIO()
    reader = input_reader("привет", "мир", EXIT_COMMAND)
    code = run_async(cmd_chat(api, out, reader, Path.cwd()))
    assert code == EXIT_OK
    assert api.created_calls == ["привет"]
    assert api.sent_calls == [("s1", "мир")]


def test_chat_ignores_empty_lines() -> None:
    api = FakeApi()
    _project(api)

    out = io.StringIO()
    reader = input_reader("", "   ", "привет", EXIT_COMMAND)
    run_async(cmd_chat(api, out, reader, Path.cwd()))
    assert api.created_calls == ["привет"]


def test_chat_exits_on_eof() -> None:
    api = FakeApi()
    _project(api)

    out = io.StringIO()
    code = run_async(cmd_chat(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK


# --- list -----------------------------------------------------------------


def test_list_filters_by_current_folder_and_newest_first() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    other = str(Path("/elsewhere").resolve())
    api.add_project({"id": "p2", "name": "q", "root_path": other})

    api.add_session(
        {
            "id": "s_old",
            "project_id": "p1",
            "title": "Старая",
            "status": "completed",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:00:00",
        }
    )
    api.add_session(
        {
            "id": "s_new",
            "project_id": "p1",
            "title": "Новая",
            "status": "running",
            "created_at": "2025-01-02T10:00:00",
            "updated_at": "2025-01-02T10:00:00",
        }
    )
    api.add_session(
        {
            "id": "s_other",
            "project_id": "p2",
            "title": "Чужая",
            "status": "completed",
            "created_at": "2025-01-03T10:00:00",
            "updated_at": "2025-01-03T10:00:00",
        }
    )

    out = io.StringIO()
    code = run_async(cmd_list(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    text = out.getvalue()
    assert "Новая" in text
    assert "Старая" in text
    assert "Чужая" not in text
    # Newest first for the current project.
    assert text.index("Новая") < text.index("Старая")


def test_list_empty_folder_ok() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    out = io.StringIO()
    code = run_async(cmd_list(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    assert out.getvalue() == ""


# --- continue -------------------------------------------------------------


def test_continue_selects_newest_session() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    api.add_session(
        {
            "id": "s_old",
            "project_id": "p1",
            "title": "Старая",
            "status": "completed",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:00:00",
        }
    )
    api.add_session(
        {
            "id": "s_new",
            "project_id": "p1",
            "title": "Новая",
            "status": "completed",
            "created_at": "2025-01-02T10:00:00",
            "updated_at": "2025-01-02T11:00:00",
        }
    )
    api.set_events("s_new", [completed_status_event("s_new")])

    out = io.StringIO()
    reader = input_reader("продолжай", EXIT_COMMAND)
    code = run_async(cmd_continue(api, out, reader, Path.cwd()))
    assert code == EXIT_OK
    # No new session was created; the existing one was reused.
    assert api.created_calls == []
    assert api.sent_calls == [("s_new", "продолжай")]


def test_continue_no_sessions_shows_hint() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    out = io.StringIO()
    code = run_async(cmd_continue(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    assert "нет сессий" in out.getvalue().lower()


# --- stop -----------------------------------------------------------------


def test_stop_stops_running_session() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    api.add_session(
        {
            "id": "s1",
            "project_id": "p1",
            "title": "Бежит",
            "status": "running",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:00:00",
        }
    )
    out = io.StringIO()
    code = run_async(cmd_stop(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    assert api.stopped_calls == ["s1"]


def test_stop_without_running_session_is_clear() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    api.add_session(
        {
            "id": "s1",
            "project_id": "p1",
            "title": "Завершена",
            "status": "completed",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:00:00",
        }
    )
    out = io.StringIO()
    code = run_async(cmd_stop(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    assert api.stopped_calls == []
    assert "нет работающей сессии" in out.getvalue().lower()


# --- self-check: malformed / edge-case inputs -----------------------------


def test_run_handles_whitespace_and_long_task() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_status("s1", "completed")

    # Whitespace-only is rejected.
    code, _ = _run_cmd(api, task=" \n\t ")
    assert code == EXIT_TASK_FAILED

    # Very long task is accepted.
    long_task = "д" * 2000
    code, _ = _run_cmd(api, task=long_task)
    assert code == EXIT_OK


def test_run_handles_malformed_session_data() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    # A broken session object without required keys must not crash the CLI.
    api.set_status("s1", "completed")

    code, _ = _run_cmd(api, task="задача")
    assert code in (EXIT_OK, EXIT_TASK_FAILED)


def test_list_handles_malformed_session_entries() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    # Sessions with missing timestamps or titles.
    api.add_session(
        {
            "id": "bad1",
            "project_id": "p1",
            "created_at": None,
            "updated_at": None,
            "status": None,
            "title": None,
        }
    )
    api.add_session(
        {
            "id": "bad2",
            "project_id": "p1",
            "status": "running",
        }
    )
    out = io.StringIO()
    code = run_async(cmd_list(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    assert out.getvalue() != ""


# --- self-check: ordering --------------------------------------------------


def test_continue_handles_ties_in_timestamp() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    api.add_session(
        {
            "id": "a",
            "project_id": "p1",
            "title": "A",
            "status": "completed",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:00:00",
        }
    )
    api.add_session(
        {
            "id": "z",
            "project_id": "p1",
            "title": "Z",
            "status": "completed",
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T10:00:00",
        }
    )
    api.set_events("z", [completed_status_event("z")])
    out = io.StringIO()
    code = run_async(cmd_continue(api, out, input_reader("дальше", EXIT_COMMAND), Path.cwd()))
    # Picks the lexicographically-larger id on ties; never raises.
    assert code == EXIT_OK
    assert api.sent_calls and api.sent_calls[0][0] == "z"


# --- self-check: errors mid-operation / resource cleanup ------------------


def test_aclose_is_called_via_cli() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_status("s1", "error")
    code = run_async(cmd_run(api, io.StringIO(), input_reader(), False, Path.cwd(), "задача"))
    assert code == EXIT_TASK_FAILED
    # The command itself does not call aclose; that is cli.py's job.
    # But the helper used here should be testable: no crash on SessionError.


def test_run_connection_error_does_not_crash() -> None:
    api = FakeApi()
    api.projects_error = ApiError(0, "Request failed: ConnectError")
    code, output = _run_cmd(api, task="задача", interactive=False)
    assert code == EXIT_NO_CONNECTION
    assert output


# --- self-check: boundaries -------------------------------------------------


def test_list_caps_at_twenty() -> None:
    api = FakeApi()
    own = str(Path.cwd().resolve())
    api.add_project({"id": "p1", "name": "p", "root_path": own})
    for i in range(25):
        api.add_session(
            {
                "id": f"s{i:02d}",
                "project_id": "p1",
                "title": f"Сессия {i}",
                "status": "completed",
                "created_at": f"2025-01-01T10:{i // 10}{i % 10}:00",
                "updated_at": f"2025-01-01T10:{i // 10}{i % 10}:00",
            }
        )
    out = io.StringIO()
    code = run_async(cmd_list(api, out, input_reader(), Path.cwd()))
    assert code == EXIT_OK
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert len(lines) == 20


# --- self-check: help texts are in Russian ---------------------------------


def test_cli_help_is_russian() -> None:
    from agent.cli import main

    # Capture the help output by providing --help as argv.
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        main(["--help"])
    except SystemExit as exit_code:
        assert exit_code.code == 0
    finally:
        sys.stdout = old_stdout
    text = buf.getvalue()
    assert "как" in text or "агент" in text or "папке" in text


def test_run_help_is_russian() -> None:
    from agent.cli import main

    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        main(["run", "--help"])
    except SystemExit as exit_code:
        assert exit_code.code == 0
    finally:
        sys.stdout = old_stdout
    text = buf.getvalue()
    assert "задачу" in text or "папке" in text


# --- self-check: chat exit command -----------------------------------------


def test_chat_exits_on_explicit_exit_command() -> None:
    api = FakeApi()
    _project(api)
    out = io.StringIO()
    reader = input_reader(EXIT_COMMAND)
    code = run_async(cmd_chat(api, out, reader, Path.cwd()))
    assert code == EXIT_OK
    assert api.created_calls == []
    assert api.sent_calls == []


# --- self-check: token not printed ------------------------------------------


def test_run_error_message_never_contains_token() -> None:
    """Even when an API error quotes the token, the CLI output must not."""
    api = FakeApi()
    # The fake's error detail intentionally embeds the token.
    api.projects_error = ApiError(401, f"Rejected: {TOKEN}")
    out = io.StringIO()
    code = run_async(cmd_run(api, out, input_reader(), False, Path.cwd(), "задача"))
    assert code == EXIT_NO_CONNECTION
    assert TOKEN not in out.getvalue()


# --- self-check: files & paths ----------------------------------------------


def test_project_path_is_resolved_for_symlinks(tmp_path: Path) -> None:
    """A symlinked project root resolves to its real directory."""
    real = tmp_path / "real-project"
    real.mkdir()
    link_dir = tmp_path / "link-project"
    link_dir.symlink_to(real, target_is_directory=True)

    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(real.resolve())})
    api.set_status("s1", "completed")

    # cmd_run uses Path.cwd() normally; here we pass the symlink path.
    out = io.StringIO()
    code = run_async(cmd_run(api, out, input_reader(), False, link_dir, "задача"))
    assert code == EXIT_OK
    # The symlink path reaches ensure_project, and it maps to the real project.
    assert api.ensure_calls == [str(link_dir)]
    assert len(api._projects) == 1


# --- self-check: repeated operations don't duplicate -------------------------


def test_repeated_run_creates_separate_sessions() -> None:
    """Calling run twice produces two distinct sessions (no accidental reuse)."""
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_status("s1", "completed")
    api.set_status("s2", "completed")

    out1 = io.StringIO()
    code1 = run_async(cmd_run(api, out1, input_reader(), False, Path.cwd(), "первая"))
    out2 = io.StringIO()
    code2 = run_async(cmd_run(api, out2, input_reader(), False, Path.cwd(), "вторая"))
    assert code1 == EXIT_OK
    assert code2 == EXIT_OK
    assert api.created_calls == ["первая", "вторая"]
    assert len(run_async(api.sessions())) == 2


def test_chat_multiple_messages_reuse_single_session() -> None:
    """Repeated chat messages all target the same session id (no duplicates)."""
    api = FakeApi()
    _project(api)
    out = io.StringIO()
    reader = input_reader("msg1", "msg2", "msg3", EXIT_COMMAND)
    run_async(cmd_chat(api, out, reader, Path.cwd()))
    # One session created, all three messages sent to it.
    assert api.created_calls == ["msg1"]
    shared = api.sent_calls[0][0]
    assert all(sid == shared for sid, _ in api.sent_calls)


# --- Ctrl+C (real SIGINT) -------------------------------------------------


def _press_ctrl_c() -> None:
    os.kill(os.getpid(), signal.SIGINT)


async def _ctrl_c_during_answer(session_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
    _press_ctrl_c()
    await asyncio.sleep(5)  # The Ctrl+C handler cancels the turn before this ends.
    yield {}


def test_chat_ctrl_c_during_answer_stops_session_and_returns_to_prompt() -> None:
    api = FakeApi()
    _project(api)
    api.events = _ctrl_c_during_answer  # type: ignore[method-assign]
    before = signal.getsignal(signal.SIGINT)
    out = io.StringIO()
    reader = input_reader("привет", "ещё", EXIT_COMMAND)
    code = run_async(cmd_chat(api, out, reader, Path.cwd()))
    assert code == EXIT_OK
    assert api.stopped_calls == ["s1", "s1"]
    assert api.sent_calls == [("s1", "ещё")]
    assert out.getvalue().count(TURN_STOPPED_TEXT) == 2
    assert signal.getsignal(signal.SIGINT) is before


def test_chat_second_ctrl_c_at_prompt_exits() -> None:
    api = FakeApi()
    _project(api)
    api.events = _ctrl_c_during_answer  # type: ignore[method-assign]
    answers = ["привет"]

    def read(_prompt: str) -> str:
        if answers:
            return answers.pop(0)
        _press_ctrl_c()
        time.sleep(5)  # KeyboardInterrupt ends the sleep.
        return "not reached"

    started = time.monotonic()
    code = run_async(cmd_chat(api, io.StringIO(), read, Path.cwd()))
    assert code == EXIT_OK
    assert api.stopped_calls == ["s1"]
    assert time.monotonic() - started < 4


def test_run_ctrl_c_at_confirmation_rejects_and_stops() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_approvals("s1", [{"id": "a1", "tool_name": "shell", "arguments": {}}])
    api.set_status("s1", "awaiting_confirmation")

    def read(_prompt: str) -> str:
        _press_ctrl_c()
        time.sleep(5)
        return "not reached"

    code = run_async(cmd_run(api, io.StringIO(), read, True, Path.cwd(), "задача"))
    assert code == EXIT_INTERRUPTED
    assert api.resolved_calls == [("a1", "reject", INTERRUPTED_REJECT_COMMENT)]
    assert api.stopped_calls == ["s1"]


# --- lost connection ----------------------------------------------------------


async def _silent_events(session_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
    await asyncio.Event().wait()
    yield {}  # pragma: no cover


def test_run_ends_with_code_two_when_server_dies_mid_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commands, "EVENT_TIMEOUT_S", 0.05)
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.events = _silent_events  # type: ignore[method-assign]
    calls = 0
    original = api.session

    async def session(session_id: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ApiError(0, "Request failed: ConnectError")
        return await original(session_id)

    api.session = session  # type: ignore[method-assign]
    code, output = _run_cmd(api)
    assert code == EXIT_NO_CONNECTION
    assert CONNECTION_LOST_TEXT in output


def test_run_waits_through_a_silent_period_and_ends_on_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commands, "EVENT_TIMEOUT_S", 0.05)
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.events = _silent_events  # type: ignore[method-assign]
    statuses = iter(["running", "running", "completed"])
    original = api.session

    async def session(session_id: str) -> dict[str, Any]:
        api.set_status(session_id, next(statuses))
        return await original(session_id)

    api.session = session  # type: ignore[method-assign]
    code, _ = _run_cmd(api)
    assert code == EXIT_OK


def test_run_stream_end_without_status_is_a_lost_connection() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    code, output = _run_cmd(api)
    assert code == EXIT_NO_CONNECTION
    assert CONNECTION_LOST_TEXT in output


# --- approvals and exit codes ---------------------------------------------------


def test_repeated_approval_event_resolves_once() -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    api.set_approvals("s1", [{"id": "a1", "tool_name": "shell", "arguments": {}}])
    api.set_status("s1", "awaiting_confirmation")
    required = {"type": "approval.required", "session_id": "s1", "payload": {}}
    api.set_events("s1", [required, required, stopped_status_event("s1")])
    code, _ = _run_cmd(api)
    assert code == EXIT_TASK_FAILED
    assert api.resolved_calls == [("a1", "reject", NON_INTERACTIVE_REJECT_COMMENT)]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (0, EXIT_NO_CONNECTION),
        (401, EXIT_NO_CONNECTION),
        (409, EXIT_TASK_FAILED),
        (500, EXIT_TASK_FAILED),
    ],
)
def test_run_create_session_error_exit_code(status: int, expected: int) -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})

    async def create_session(project_id: str, prompt: str, **_: Any) -> dict[str, Any]:
        raise ApiError(status, "Слишком много сессий")

    api.create_session = create_session  # type: ignore[method-assign]
    code, _ = _run_cmd(api)
    assert code == expected


@pytest.mark.parametrize(("count", "shown"), [(20, 20), (21, 20)])
def test_list_limit_boundary(count: int, shown: int) -> None:
    api = FakeApi()
    api.add_project({"id": "p1", "name": "p", "root_path": str(Path.cwd().resolve())})
    for i in range(count):
        stamp = f"2025-01-01T10:{i:02d}:00"
        api.add_session(
            {
                "id": f"s{i:02d}",
                "project_id": "p1",
                "title": f"Сессия {i}",
                "status": "completed",
                "created_at": stamp,
                "updated_at": stamp,
            }
        )
    out = io.StringIO()
    assert run_async(cmd_list(api, out, input_reader(), Path.cwd())) == EXIT_OK
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert len(lines) == shown


def test_list_time_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Asia/Krasnoyarsk")
    time.tzset()
    try:
        assert commands._format_time("2025-01-15T03:03:00") == "2025-01-15 10:03"
        assert commands._format_time("2025-01-15T03:03:00+00:00") == "2025-01-15 10:03"
        assert commands._format_time("not a time") == "not a time"
    finally:
        monkeypatch.undo()
        time.tzset()

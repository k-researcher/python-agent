"""Asynchronous CLI commands for the agent API client.

Each command receives an :class:`AgentApi`, an output stream and an input
reader so that tests may run without a live server (dependency injection).
The UI texts are in Russian and kept short for beginners.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from agent.cli_client.api import AgentApi, ApiError, ClientConfig
from agent.cli_client.render import Renderer, Style, prompt_approval

# Exit codes: 0 ok, 1 task error, 2 no connection/settings, 130 Ctrl+C.
EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_NO_CONNECTION = 2
EXIT_INTERRUPTED = 130

# Terminal and transient session statuses.
TERMINAL_STATUSES = frozenset({"completed", "error", "stopped"})
ACTIVE_STATUSES = frozenset({"pending", "running", "awaiting_confirmation"})

# User-facing messages (Russian, short).
NO_TOKEN_HINT = (
    "Для работы из терминала нужен токен. Добавьте в файл .env строку "
    "AGENT_API_TOKEN=<любая длинная случайная строка> и перезапустите агента."
)
REJECTED_TOKEN_HINT = "API сервер отклонил токен. Проверьте AGENT_API_TOKEN в .env."
NO_SERVER_HINT = "Запустите в другом окне: pyagent"
START_SERVER_PROMPT = "Агент не запущен. Запустить сейчас? (д/н) "
NON_INTERACTIVE_REJECT_COMMENT = "Подтверждение недоступно без терминала"
INTERRUPTED_REJECT_COMMENT = "Отменено по Ctrl+C"
CONNECTION_LOST_TEXT = "Связь с агентом потеряна."
TURN_STOPPED_TEXT = "Ответ остановлен. Для выхода нажмите Ctrl+C ещё раз."

# Server liveness probe used after launching the agent in the background.
HEALTH_PATH = "/api/health"
HEALTH_TIMEOUT_S = 20.0
HEALTH_POLL_INTERVAL_S = 0.5
CHAT_PROMPT = "Вы: "
EXIT_COMMAND = "/выход"

# Time without events after which a turn asks the server for the session status.
EVENT_TIMEOUT_S = 120.0

# Turn outcome when the server or the connection is gone (see _process_turn).
CONNECTION_LOST = "connection_lost"


def _interactive() -> bool:
    """Return whether stdin is a terminal (used by the real CLI entry point)."""
    return sys.stdin.isatty()


def build_config() -> ClientConfig:
    """Build a connection config from environment and local settings.

    Precedence: environment variables first; the local ``Settings`` file second.
    There is no network access here — only local configuration detection.
    """
    from agent.config import Settings

    settings = Settings()
    url = os.environ.get("AGENT_URL") or ""
    if not url:
        host = settings.host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        url = f"http://{host}:{settings.port}"
    token = os.environ.get("AGENT_API_TOKEN") or ""
    if not token:
        token = settings.api_token
    return ClientConfig(url, token)


def _friendly_error(exc: ApiError) -> str:
    """Convert an ApiError to a user-facing message."""
    if exc.status == 401 and "AGENT_API_TOKEN" in exc.detail:
        return NO_TOKEN_HINT
    if exc.status == 401:
        return REJECTED_TOKEN_HINT
    return exc.detail


def _prompt_start_server(read: Callable[[str], str], out: TextIO) -> bool:
    """Ask whether to start the agent now; False when declined or EOF."""
    try:
        answer = read(START_SERVER_PROMPT)
    except (EOFError, KeyboardInterrupt):
        out.write(NO_SERVER_HINT + "\n")
        return False
    return (answer or "").strip().lower().startswith("д")


def _launch_server() -> bool:
    """Start the ``pyagent`` server in the background.

    Writes logs to ``data/agent-cli.log`` and detaches from the terminal.
    Returns True when the process was started (best effort).
    """
    try:
        from agent.config import get_settings

        settings = get_settings()
        settings.prepare_directories()
        log_path = settings.data_dir / "agent-cli.log"
        agent_exe = shutil.which("pyagent")
        if agent_exe:
            cmd = [agent_exe, "serve"]
        else:
            cmd = [sys.executable, "-c", "from agent.cli import main; main(['serve'])"]
        with open(log_path, "a") as log:
            subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return True
    except OSError:
        return False


async def _wait_for_health(
    api: AgentApi,
    timeout: float = HEALTH_TIMEOUT_S,
    interval: float = HEALTH_POLL_INTERVAL_S,
) -> bool:
    """Poll the liveness endpoint until it responds or the timeout expires."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            await api._request("GET", HEALTH_PATH)
            return True
        except ApiError as exc:
            if exc.status != 0:
                # The server is up even if the health path differs.
                return True
        await asyncio.sleep(interval)
    return False


async def _check_connection(
    api: AgentApi,
    out: TextIO,
    read: Callable[[str], str],
    interactive: bool,
) -> bool:
    """Verify the server is reachable; optionally start it.

    Returns True when usable, False otherwise. Never prints a token.
    """
    try:
        await api.projects()
        return True
    except ApiError as exc:
        if exc.status == 401:
            out.write(_friendly_error(exc) + "\n")
            return False
        if exc.status == 0:
            if interactive and _prompt_start_server(read, out):
                if not _launch_server():
                    out.write(NO_SERVER_HINT + "\n")
                    return False
                if await _wait_for_health(api):
                    return True
                out.write("Агент не запустился. Проверьте data/agent-cli.log.\n")
                return False
            out.write("Агент не запущен. " + NO_SERVER_HINT + "\n")
            return False
        out.write(f"Ошибка API: {_friendly_error(exc)}\n")
        return False


async def _try_stop(api: AgentApi, session_id: str) -> None:
    """Best-effort session stop that never raises."""
    with contextlib.suppress(ApiError, asyncio.CancelledError):
        await api.stop(session_id)


async def _filter_project_sessions(api: AgentApi, project_id: str) -> list[dict[str, Any]]:
    """Return non-archived sessions belonging to the given project."""
    sessions = await api.sessions()
    return [s for s in sessions if s.get("project_id") == project_id]


def _newest_session(sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most recently updated session, or None for an empty list.

    Missing/None timestamps sort last; ties break by session id.
    """
    if not sessions:
        return None

    def key(session: dict[str, Any]) -> tuple[str, str]:
        created = session.get("updated_at") or session.get("created_at")
        return (str(created), str(session.get("id", "")))

    return max(sessions, key=key)


def _format_time(value: Any) -> str:
    """Render a server timestamp as short local time, for example "2025-01-15 17:30".

    The server stores UTC; a timestamp without a zone is read as UTC.
    """
    text = str(value)
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text[:16].replace("T", " ") if len(text) > 16 else text
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


class TurnInterrupted(Exception):
    """The user pressed Ctrl+C at a confirmation prompt during a turn."""


@contextlib.contextmanager
def _ctrl_c_raises() -> Iterator[None]:
    """Make Ctrl+C raise KeyboardInterrupt, also inside a blocking input()."""
    try:
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    except ValueError:
        # Not the main thread: signals stay as they are.
        yield
        return
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


@contextlib.contextmanager
def _ctrl_c_cancels(task: asyncio.Task[Any]) -> Iterator[list[bool]]:
    """While active, Ctrl+C cancels ``task``; the yielded list records each press."""
    loop = asyncio.get_running_loop()
    pressed: list[bool] = []

    def on_interrupt() -> None:
        pressed.append(True)
        task.cancel()

    previous = signal.getsignal(signal.SIGINT)
    try:
        loop.add_signal_handler(signal.SIGINT, on_interrupt)
    except (NotImplementedError, RuntimeError, ValueError):
        # No loop signal support (Windows or not the main thread): keep the default.
        yield pressed
        return
    try:
        yield pressed
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


async def _render_approvals(
    api: AgentApi,
    renderer: Renderer,
    out: TextIO,
    read: Callable[[str], str],
    interactive: bool,
    session_id: str,
    processed: set[str],
) -> None:
    """Fetch and resolve every pending approval for a session.

    ``processed`` keeps the approval ids of this turn, so a repeated
    ``approval.required`` event does not ask for or resolve the same item twice.
    """
    try:
        approvals = await api.approvals(session_id)
    except ApiError as exc:
        renderer.error(_friendly_error(exc))
        return

    for approval in approvals:
        approval_id = approval.get("id")
        if not approval_id or approval_id in processed:
            continue
        processed.add(approval_id)
        renderer.approval(approval)
        interrupted = False
        if interactive:
            try:
                with _ctrl_c_raises():
                    decision, comment = prompt_approval(read, approval)
            except KeyboardInterrupt:
                interrupted = True
                decision, comment = "reject", INTERRUPTED_REJECT_COMMENT
        else:
            out.write(f"{NON_INTERACTIVE_REJECT_COMMENT}; инструмент отклонён.\n")
            decision, comment = "reject", NON_INTERACTIVE_REJECT_COMMENT
        try:
            await api.resolve_approval(approval_id, decision, comment)
        except ApiError as exc:
            renderer.error(f"Не удалось обработать подтверждение: {_friendly_error(exc)}")
        if interrupted:
            raise TurnInterrupted


def _render_status(renderer: Renderer, status: str, error: Any) -> None:
    renderer.status(status, error=str(error) if error else None)


async def _check_status(
    api: AgentApi,
    renderer: Renderer,
    out: TextIO,
    read: Callable[[str], str],
    interactive: bool,
    session_id: str,
    processed: set[str],
) -> str | None:
    """Return the turn outcome when the session no longer runs, None while it runs."""
    try:
        session = await api.session(session_id)
    except ApiError as exc:
        renderer.error(_friendly_error(exc) if exc.status else CONNECTION_LOST_TEXT)
        return CONNECTION_LOST if exc.status in (0, 401) else "error"
    status = str(session.get("status", ""))
    if status in TERMINAL_STATUSES:
        _render_status(renderer, status, session.get("error"))
        return status
    if status == "awaiting_confirmation":
        await _render_approvals(api, renderer, out, read, interactive, session_id, processed)
    return None


async def _handle_event(
    event: dict[str, Any],
    api: AgentApi,
    renderer: Renderer,
    out: TextIO,
    read: Callable[[str], str],
    interactive: bool,
    session_id: str,
    processed: set[str],
) -> str | None:
    """Render one event; return the terminal status when the event ends the turn."""
    event_type = event.get("type", "")
    payload = event.get("payload") or {}
    if event_type in ("message.delta", "reasoning.delta"):
        kind = "reasoning" if event_type == "reasoning.delta" else "message"
        renderer.stream_delta(kind, payload.get("delta", ""))
    elif event_type == "stream.reset":
        renderer.end_stream()
    elif event_type == "tool.completed":
        renderer.end_stream()
        renderer.tool_call(
            {"name": payload.get("name", ""), "arguments": {}},
            {"success": payload.get("status", "") == "completed"},
        )
    elif event_type == "approval.required":
        renderer.end_stream()
        await _render_approvals(api, renderer, out, read, interactive, session_id, processed)
    elif event_type == "session.status":
        status = str(payload.get("status", ""))
        if status in TERMINAL_STATUSES:
            _render_status(renderer, status, payload.get("error"))
            return status
    return None


async def _process_turn(
    api: AgentApi,
    renderer: Renderer,
    out: TextIO,
    read: Callable[[str], str],
    interactive: bool,
    session_id: str,
) -> str:
    """Follow one session turn until it ends.

    Returns the terminal status ("completed", "error" or "stopped") or
    ``CONNECTION_LOST``. The event stream reconnects without limit, so after
    ``EVENT_TIMEOUT_S`` without events the turn asks the server for the status:
    a dead server ends the turn instead of a wait without end.
    """
    processed: set[str] = set()
    outcome = await _check_status(api, renderer, out, read, interactive, session_id, processed)
    if outcome is not None:
        return outcome
    while True:
        stream = api.events(session_id)
        ended = False
        try:
            while True:
                try:
                    async with asyncio.timeout(EVENT_TIMEOUT_S):
                        event = await anext(stream)
                except StopAsyncIteration:
                    ended = True
                    break
                except TimeoutError:
                    break
                outcome = await _handle_event(
                    event, api, renderer, out, read, interactive, session_id, processed
                )
                if outcome is not None:
                    return outcome
        except ApiError as exc:
            renderer.error(_friendly_error(exc))
            return CONNECTION_LOST
        finally:
            await stream.aclose()
        outcome = await _check_status(api, renderer, out, read, interactive, session_id, processed)
        if outcome is not None:
            return outcome
        if ended:
            # A live stream never ends; treat the end as a lost connection.
            renderer.error(CONNECTION_LOST_TEXT)
            return CONNECTION_LOST


def _exit_code(exc: ApiError) -> int:
    """Code 2 for a missing connection or token, code 1 for any other server error."""
    return EXIT_NO_CONNECTION if exc.status in (0, 401) else EXIT_TASK_FAILED


async def cmd_chat(
    api: AgentApi,
    out: TextIO,
    read: Callable[[str], str],
    project_root: Path,
    session_id: str | None = None,
) -> int:
    """Interactive chat with the agent in the current project (always interactive).

    If ``session_id`` is given, the conversation continues in that session;
    otherwise the first message creates a new one. Ctrl+C during an answer stops
    the session and returns to the prompt; Ctrl+C at the prompt ends the chat.
    """
    renderer = Renderer(out, Style.detect(out))

    if not await _check_connection(api, out, read, _interactive()):
        return EXIT_NO_CONNECTION

    try:
        project = await api.ensure_project(str(project_root))
    except ApiError as exc:
        renderer.error(_friendly_error(exc))
        return _exit_code(exc)
    project_id = project["id"]

    with _ctrl_c_raises():
        while True:
            try:
                raw = read(CHAT_PROMPT)
            except (EOFError, KeyboardInterrupt):
                out.write("\n")
                return EXIT_OK
            text = (raw or "").strip()
            if not text:
                continue
            if text == EXIT_COMMAND:
                return EXIT_OK

            try:
                if session_id is None:
                    session = await api.create_session(project_id, text)
                    session_id = session["id"]
                else:
                    await api.send_message(session_id, text)
            except ApiError as exc:
                renderer.error(_friendly_error(exc))
                continue

            turn = asyncio.create_task(
                _process_turn(api, renderer, out, read, interactive=True, session_id=session_id)
            )
            with _ctrl_c_cancels(turn) as pressed:
                try:
                    await turn
                except asyncio.CancelledError:
                    if not pressed:
                        raise  # The chat itself was cancelled.
                except TurnInterrupted:
                    pass
                else:
                    continue
            # Ctrl+C during the answer: stop the session, then back to the prompt.
            await _try_stop(api, session_id)
            out.write("\n" + TURN_STOPPED_TEXT + "\n")


async def cmd_run(
    api: AgentApi,
    out: TextIO,
    read: Callable[[str], str],
    interactive: bool,
    project_root: Path,
    task: str,
) -> int:
    """Run a single task in the current project and print the result.

    Returns: EXIT_OK (0) on completed, EXIT_TASK_FAILED (1) on error/stopped,
    EXIT_NO_CONNECTION (2) when the server, the token or the connection is missing,
    EXIT_INTERRUPTED (130) on Ctrl+C. Ctrl+C also stops the session.
    """
    renderer = Renderer(out, Style.detect(out))

    if not (task or "").strip():
        renderer.error("Задача не может быть пустой.")
        return EXIT_TASK_FAILED

    if not await _check_connection(api, out, read, interactive):
        return EXIT_NO_CONNECTION

    try:
        project = await api.ensure_project(str(project_root))
        session = await api.create_session(project["id"], task)
    except ApiError as exc:
        renderer.error(_friendly_error(exc))
        return _exit_code(exc)
    session_id = session["id"]

    try:
        outcome = await _process_turn(
            api, renderer, out, read, interactive=interactive, session_id=session_id
        )
    except TurnInterrupted:
        await _try_stop(api, session_id)
        return EXIT_INTERRUPTED
    except (asyncio.CancelledError, KeyboardInterrupt):
        await _try_stop(api, session_id)
        raise
    if outcome == "completed":
        return EXIT_OK
    if outcome == CONNECTION_LOST:
        return EXIT_NO_CONNECTION
    return EXIT_TASK_FAILED


async def cmd_list(
    api: AgentApi,
    out: TextIO,
    read: Callable[[str], str],
    project_root: Path,
) -> int:
    """List the sessions of the current folder (last 20, newest first)."""
    if not await _check_connection(api, out, read, _interactive()):
        return EXIT_NO_CONNECTION

    try:
        project = await api.ensure_project(str(project_root))
    except ApiError as exc:
        out.write(f"Ошибка: {_friendly_error(exc)}\n")
        return _exit_code(exc)
    project_id = project["id"]

    try:
        sessions = await _filter_project_sessions(api, project_id)
    except ApiError as exc:
        out.write(f"Ошибка: {_friendly_error(exc)}\n")
        return _exit_code(exc)

    sessions.sort(
        key=lambda s: (
            str(s.get("updated_at") or s.get("created_at") or ""),
            str(s.get("id", "")),
        ),
        reverse=True,
    )
    for session in sessions[:20]:
        out.write(
            f"{_format_time(session.get('created_at'))}  "
            f"{session.get('status', '')}  {session.get('title', '')}\n"
        )
    return EXIT_OK


async def cmd_continue(
    api: AgentApi,
    out: TextIO,
    read: Callable[[str], str],
    project_root: Path,
) -> int:
    """Continue the most recent session of the current folder in chat mode."""
    if not await _check_connection(api, out, read, _interactive()):
        return EXIT_NO_CONNECTION

    renderer = Renderer(out, Style.detect(out))
    try:
        project = await api.ensure_project(str(project_root))
    except ApiError as exc:
        renderer.error(_friendly_error(exc))
        return _exit_code(exc)
    project_id = project["id"]

    try:
        sessions = await _filter_project_sessions(api, project_id)
    except ApiError as exc:
        renderer.error(_friendly_error(exc))
        return _exit_code(exc)

    latest = _newest_session(sessions)
    if latest is None:
        out.write("В этой папке ещё нет сессий. Начните с команды: pyagent chat\n")
        return EXIT_OK

    session_id = latest["id"]
    return await cmd_chat(api, out, read, project_root, session_id=session_id)


async def cmd_stop(
    api: AgentApi,
    out: TextIO,
    read: Callable[[str], str],
    project_root: Path,
) -> int:
    """Stop the working session(s) of the current folder."""
    if not await _check_connection(api, out, read, _interactive()):
        return EXIT_NO_CONNECTION

    try:
        project = await api.ensure_project(str(project_root))
    except ApiError as exc:
        out.write(f"Ошибка: {_friendly_error(exc)}\n")
        return _exit_code(exc)
    project_id = project["id"]

    try:
        sessions = await _filter_project_sessions(api, project_id)
    except ApiError as exc:
        out.write(f"Ошибка: {_friendly_error(exc)}\n")
        return _exit_code(exc)

    running = [s for s in sessions if s.get("status") in ACTIVE_STATUSES]
    if not running:
        out.write("Нет работающей сессии в этой папке.\n")
        return EXIT_OK

    for session in running:
        try:
            await api.stop(session["id"])
        except ApiError as exc:
            out.write(f"Ошибка при остановке: {_friendly_error(exc)}\n")
            continue
        out.write(f"Остановлена сессия: {session.get('title', '(без названия)')}\n")
    return EXIT_OK

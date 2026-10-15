"""Terminal renderer for the asynchronous CLI agent client.

Formats streamed deltas, messages, tool calls, approvals and status lines.
ANSI colors are used only when the output is a TTY and NO_COLOR is not set.
UI texts are in Russian.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

_DIM = 2
_RED = 31
_GREEN = 32
_YELLOW = 33
_RESET = 0

_SHELL_TOOLS = frozenset({"shell"})
_FILE_TOOLS = frozenset(
    {
        "list_dir",
        "read_file",
        "read_many",
        "write_file",
        "edit_file",
        "multi_edit",
        "remove_file",
    }
)
_PATTERN_TOOLS = frozenset({"glob", "search_files"})

MAX_APPROVAL_LINES = 40
MAX_JSON_ARG_CHARS = 60


@dataclass(frozen=True)
class Style:
    """Describe whether the stream accepts ANSI escape sequences."""

    color: bool

    @classmethod
    def detect(cls, stream: TextIO) -> Style:
        """Enable color for a TTY unless NO_COLOR is present."""
        no_color = os.environ.get("NO_COLOR") is not None
        return cls(color=stream.isatty() and not no_color)


def _coerce_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
    """Return a dict of arguments and a JSON fallback for non-dict input."""
    if isinstance(arguments, str):
        try:
            parsed: Any = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {}, arguments
        if isinstance(parsed, dict):
            return parsed, None
        return {}, arguments
    if isinstance(arguments, dict):
        return arguments, None
    return {}, str(arguments)


def _short_arg(name: str, arguments: Any) -> str:
    """Render a compact human-readable argument for a tool call."""
    data, fallback = _coerce_arguments(arguments)
    if fallback is not None:
        return fallback[:MAX_JSON_ARG_CHARS]
    if name in _SHELL_TOOLS:
        value = data.get("command")
        if isinstance(value, str):
            return value
    if name in _FILE_TOOLS:
        value = data.get("path")
        if value is not None:
            return str(value)
    if name in _PATTERN_TOOLS:
        value = data.get("pattern")
        if isinstance(value, str):
            return value
    try:
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return ""
    return encoded[:MAX_JSON_ARG_CHARS]


class Renderer:
    """Write human-readable terminal lines to a text stream."""

    _ROLE_LABELS = {
        "user": "Пользователь",
        "assistant": "Ассистент",
        "tool": "Инструмент",
        "summary": "Сводка",
    }

    def __init__(self, out: TextIO, style: Style) -> None:
        self._out = out
        self._style = style
        self._stream_open = False

    def _stylize(self, text: str, *codes: int) -> str:
        if not codes or not self._style.color:
            return text
        sequence = ";".join(str(code) for code in codes)
        return f"\x1b[{sequence}m{text}\x1b[{_RESET}m"

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def stream_delta(self, kind: str, text: str) -> None:
        """Stream a partial message or reasoning fragment."""
        if kind == "reasoning":
            prefix = self._stylize("· ", _DIM)
            body = self._stylize(text, _DIM)
            self._write(prefix + body)
        else:
            self._write(text)
        self._stream_open = True

    def end_stream(self) -> None:
        """Close an open stream line with a newline."""
        if self._stream_open:
            self._write("\n")
            self._stream_open = False

    def message(self, message: dict[str, Any]) -> None:
        """Render a stored message by role."""
        self.end_stream()
        role = str(message.get("role", ""))
        content = message.get("content")
        text = str(content) if content is not None else ""
        label = self._ROLE_LABELS.get(role, role or "Сообщение")
        self._write(f"{label}: {text}\n")
        self._stream_open = False

    def tool_call(self, call: dict[str, Any], result: dict[str, Any] | None) -> None:
        """Render one tool invocation with its outcome symbol."""
        self.end_stream()
        # Messages carry OpenAI-style calls ({"function": {...}}); tool call rows are flat.
        function = call.get("function")
        source: dict[str, Any] = function if isinstance(function, dict) else call
        name = str(source.get("name") or "")
        short = _short_arg(name, source.get("arguments"))
        if result is None:
            symbol = self._stylize("…", _YELLOW)
        elif _is_success(result):
            symbol = self._stylize("✓", _GREEN)
        else:
            symbol = self._stylize("✗", _RED)
        self._write(f"▸ {name}  {short}  {symbol}\n")
        self._stream_open = False

    def approval(self, approval: dict[str, Any]) -> None:
        """Render a request for tool approval including pretty arguments."""
        self.end_stream()
        name = str(approval.get("tool", approval.get("tool_name", "")))
        risk = str(approval.get("risk_level", ""))
        self._write(f"Запрос инструмента: {name}")
        if risk:
            self._write(f" (риск: {risk})")
        self._write("\n")
        arguments = approval.get("arguments")
        for line in _pretty_lines(arguments):
            self._write(line + "\n")
        self._stream_open = False

    def status(self, status: str, error: str | None = None) -> None:
        """Render a status line, optionally with an error detail."""
        self.end_stream()
        self._write(f"Статус: {status}\n")
        if error:
            self._write(self._stylize(f"Ошибка: {error}\n", _RED))
        self._stream_open = False

    def error(self, text: str) -> None:
        """Render an error message."""
        self.end_stream()
        self._write(self._stylize(f"Ошибка: {text}\n", _RED))
        self._stream_open = False


def _is_success(result: dict[str, Any]) -> bool:
    """Return whether the tool result reports success."""
    value = result.get("success", result.get("ok"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "ok", "success"}
    return False


def _pretty_lines(arguments: Any) -> list[str]:
    """Return at most MAX_APPROVAL_LINES lines of readable arguments."""
    if isinstance(arguments, (dict, list)):
        try:
            lines = json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            lines = str(arguments)
    else:
        lines = str(arguments)
    split = lines.splitlines()
    if len(split) <= MAX_APPROVAL_LINES:
        return split
    return split[:MAX_APPROVAL_LINES] + [f"… (ещё {len(split) - MAX_APPROVAL_LINES} строк)"]


PROMPT_TEXT = "Разрешить? д — да, н — нет, к — нет и написать почему: "
COMMENT_TEXT = "Почему нет (агент увидит этот текст): "
_YES = frozenset({"д", "да", "y", "yes"})
_NO = frozenset({"н", "нет", "n", "no"})
_COMMENT = frozenset({"к", "c", "comment"})


def prompt_approval(read: Callable[[str], str], approval: dict[str, Any]) -> tuple[str, str | None]:
    """Ask the user to approve a tool call; ask again until the answer is clear."""
    while True:
        try:
            answer = (read(PROMPT_TEXT) or "").strip().lower()
        except EOFError:
            return ("reject", None)  # Ctrl+D: refuse; never hang and never approve
        if answer in _YES:
            return ("approve", None)
        if answer in _NO:
            return ("reject", None)
        if answer in _COMMENT:
            try:
                comment = (read(COMMENT_TEXT) or "").strip() or None
            except EOFError:
                comment = None
            return ("reject", comment)

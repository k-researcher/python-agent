"""Unit tests for the CLI terminal renderer."""

from __future__ import annotations

import io

import pytest

from agent.cli_client.render import (
    COMMENT_TEXT,
    PROMPT_TEXT,
    Renderer,
    Style,
    _short_arg,
    prompt_approval,
)


class FakeTTY(io.StringIO):
    """StringIO that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True


def renderer(color: bool) -> tuple[Renderer, io.StringIO]:
    buf = io.StringIO()
    return Renderer(buf, Style(color=color)), buf


def test_style_detect_no_tty() -> None:
    style = Style.detect(io.StringIO())
    assert style.color is False


def test_style_detect_tty_without_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert Style.detect(FakeTTY()).color is True


def test_style_detect_no_color_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert Style.detect(FakeTTY()).color is False


def test_no_color_no_ansi() -> None:
    r, buf = renderer(False)
    r.stream_delta("reasoning", "hmm")
    r.end_stream()
    r.error("bad")
    r.tool_call({"name": "shell", "arguments": {"command": "ls"}}, None)
    assert "\x1b[" not in buf.getvalue()


def test_color_adds_ansi() -> None:
    r, buf = renderer(True)
    r.error("boom")
    assert "\x1b[" in buf.getvalue()


def test_short_arg_shell() -> None:
    assert _short_arg("shell", {"command": "git status"}) == "git status"


def test_short_arg_file_path() -> None:
    assert _short_arg("read_file", {"path": "src/a.py"}) == "src/a.py"
    assert _short_arg("write_file", {"path": "notes.txt"}) == "notes.txt"


def test_short_arg_pattern() -> None:
    assert _short_arg("glob", {"pattern": "**/*.py"}) == "**/*.py"
    assert _short_arg("search_files", {"pattern": "TODO"}) == "TODO"


def test_short_arg_fallback_json() -> None:
    args = {"db": "app", "kind": "user"}
    short = _short_arg("db_select", args)
    assert short == '{"db": "app", "kind": "user"}'
    assert len(short) <= 60


def test_short_arg_fallback_truncated() -> None:
    args = {"payload": "x" * 200}
    short = _short_arg("db_select", args)
    assert len(short) == 60
    assert short.endswith("...") is False


def test_short_arg_string_arguments() -> None:
    assert _short_arg("shell", '{"command": "ls"}') == "ls"
    assert _short_arg("db_select", '{"k": "v"}') == '{"k": "v"}'


def test_short_arg_non_mapping() -> None:
    assert _short_arg("db_select", [1, 2, 3]) == "[1, 2, 3]"


def test_stream_delta_reasoning_prefix() -> None:
    r, buf = renderer(False)
    r.stream_delta("reasoning", "think")
    r.end_stream()
    assert buf.getvalue() == "· think\n"


def test_stream_delta_message_no_prefix() -> None:
    r, buf = renderer(False)
    r.stream_delta("message", "hello ")
    r.stream_delta("message", "world")
    r.end_stream()
    assert buf.getvalue() == "hello world\n"


def test_end_stream_without_open_line_adds_nothing() -> None:
    r, buf = renderer(False)
    r.end_stream()
    assert buf.getvalue() == ""


def test_message_roles() -> None:
    r, buf = renderer(False)
    r.message({"role": "user", "content": "hi"})
    r.message({"role": "assistant", "content": "yo"})
    r.message({"role": "tool", "content": "ok"})
    r.message({"role": "summary", "content": "sum"})
    assert buf.getvalue() == ("Пользователь: hi\nАссистент: yo\nИнструмент: ok\nСводка: sum\n")


def test_message_unknown_role_falls_back() -> None:
    r, buf = renderer(False)
    r.message({"role": "system", "content": "x"})
    assert buf.getvalue() == "system: x\n"


def test_message_missing_content() -> None:
    r, buf = renderer(False)
    r.message({"role": "user"})
    assert buf.getvalue() == "Пользователь: \n"


def test_tool_call_pending() -> None:
    r, buf = renderer(False)
    r.tool_call({"name": "shell", "arguments": {"command": "ls"}}, None)
    assert buf.getvalue() == "▸ shell  ls  …\n"


def test_tool_call_success() -> None:
    r, buf = renderer(False)
    r.tool_call({"name": "shell", "arguments": {"command": "ls"}}, {"success": True})
    assert buf.getvalue() == "▸ shell  ls  ✓\n"


def test_tool_call_failure() -> None:
    r, buf = renderer(False)
    r.tool_call({"name": "shell", "arguments": {"command": "ls"}}, {"success": False})
    assert buf.getvalue() == "▸ shell  ls  ✗\n"


def test_tool_call_colors_symbols() -> None:
    r, buf = renderer(True)
    r.tool_call({"name": "shell", "arguments": {"command": "ls"}}, {"success": True})
    assert "\x1b[32m" in buf.getvalue()
    assert "✓" in buf.getvalue()


def test_approval_basic() -> None:
    r, buf = renderer(False)
    r.approval(
        {
            "tool": "shell",
            "risk_level": "process_execution",
            "arguments": {"command": "ls"},
        }
    )
    value = buf.getvalue()
    assert "Запрос инструмента: shell" in value
    assert "риск: process_execution" in value
    assert '"command": "ls"' in value


def test_approval_truncates_long_arguments() -> None:
    long_args = {"items": [f"value-{i}" for i in range(100)]}
    r, buf = renderer(False)
    r.approval({"tool": "shell", "risk_level": "low", "arguments": long_args})
    lines = buf.getvalue().splitlines()
    # 1 header + up to 40 JSON lines + 1 truncation marker.
    assert len(lines) == 42
    assert "ещё" in lines[-1]


def test_approval_short_arguments_not_truncated() -> None:
    r, buf = renderer(False)
    r.approval({"tool": "shell", "risk_level": "low", "arguments": {"command": "ls"}})
    lines = buf.getvalue().splitlines()
    assert len(lines) == 4  # header + 3 pretty-printed JSON lines


def test_status_with_and_without_error() -> None:
    r, buf = renderer(False)
    r.status("running")
    r.status("failed", error="bad request")
    value = buf.getvalue()
    assert "Статус: running\n" in value
    assert "Статус: failed\n" in value
    assert "Ошибка: bad request\n" in value


def test_error() -> None:
    r, buf = renderer(False)
    r.error("boom")
    assert buf.getvalue() == "Ошибка: boom\n"


def test_stream_delta_then_tool_call_closes_line() -> None:
    r, buf = renderer(False)
    r.stream_delta("message", "processing")
    r.tool_call({"name": "shell", "arguments": {"command": "ls"}}, None)
    assert buf.getvalue() == "processing\n▸ shell  ls  …\n"


def test_prompt_approval_yes() -> None:
    answers = iter(["y"])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("approve", None)


def test_prompt_approval_no() -> None:
    answers = iter(["n"])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("reject", None)


def test_prompt_approval_comment() -> None:
    answers = iter(["c", "too risky"])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("reject", "too risky")


def test_prompt_approval_comment_empty() -> None:
    answers = iter(["c", "   "])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("reject", None)


def test_prompt_approval_empty_input_repeats() -> None:
    prompts_seen: list[str] = []
    answers = iter(["", "   ", "y"])

    def read(prompt: str) -> str:
        prompts_seen.append(prompt)
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("approve", None)
    assert prompts_seen.count(PROMPT_TEXT) == 3


def test_prompt_approval_invalid_repeats() -> None:
    answers = iter(["z", "n"])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("reject", None)


def test_prompt_approval_case_insensitive() -> None:
    answers = iter(["Y"])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("approve", None)


def test_prompt_approval_uppercase_comment() -> None:
    answers = iter(["C", "comment"])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("reject", "comment")


def test_prompt_approval_strips_comment() -> None:
    answers = iter(["c", "  spaced  "])

    def read(_prompt: str) -> str:
        return next(answers)

    assert prompt_approval(read, {"tool": "shell"}) == ("reject", "spaced")


def test_prompt_approval_prompts() -> None:
    prompts: list[str] = []
    answers = iter(["y"])

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    prompt_approval(read, {"tool": "shell"})
    assert prompts == [PROMPT_TEXT]


def test_prompt_approval_comment_prompt() -> None:
    prompts: list[str] = []
    answers = iter(["c", "why"])

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    prompt_approval(read, {"tool": "shell"})
    assert prompts == [PROMPT_TEXT, COMMENT_TEXT]


def test_tool_call_accepts_openai_message_shape() -> None:
    out = io.StringIO()
    renderer = Renderer(out, Style(color=False))
    call = {"id": "c1", "function": {"name": "shell", "arguments": '{"command": "pytest -q"}'}}
    renderer.tool_call(call, {"success": True})
    assert "▸ shell  pytest -q  ✓" in out.getvalue()


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["д"], ("approve", None)),
        (["да"], ("approve", None)),
        (["y"], ("approve", None)),
        (["н"], ("reject", None)),
        (["к", "опасно"], ("reject", "опасно")),
        (["?", "д"], ("approve", None)),
    ],
)
def test_prompt_accepts_russian_and_english_answers(
    answers: list[str], expected: tuple[str, str | None]
) -> None:
    replies = iter(answers)
    assert prompt_approval(lambda _prompt: next(replies), {}) == expected


def test_message_closes_an_open_stream_line() -> None:
    out = io.StringIO()
    renderer = Renderer(out, Style(color=False))
    renderer.stream_delta("message", "processing")
    renderer.message({"role": "user", "content": "hi"})
    assert out.getvalue().startswith("processing\n")


def test_prompt_rejects_on_end_of_input() -> None:
    def read(_prompt: str) -> str:
        raise EOFError

    assert prompt_approval(read, {}) == ("reject", None)

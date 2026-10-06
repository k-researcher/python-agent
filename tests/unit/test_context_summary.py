import json
import re

from agent.context_summary import (
    MAX_SUMMARY_CALLS,
    SECTIONS,
    SUMMARY_LABEL,
    SUMMARY_MAX_TOKENS,
    SUMMARY_VERSION,
    SummaryCheckpoint,
    build_summary_request,
    deterministic_summary,
    normalize_summary,
    render_summary_item,
    source_hash,
)


def _user(content: str) -> dict[str, object]:
    return {"role": "user", "content": content}


def _assistant(name: str, arguments: str, call_id: str = "call_1") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def _tool(content: str, call_id: str) -> dict[str, object]:
    return {"role": "tool", "content": content, "tool_call_id": call_id}


def test_constants() -> None:
    assert SUMMARY_VERSION == 1
    assert SUMMARY_MAX_TOKENS == 1024
    assert MAX_SUMMARY_CALLS == 4
    assert SUMMARY_LABEL
    assert SECTIONS == (
        "Goal",
        "Constraints",
        "Decisions",
        "Changed paths",
        "Check results",
        "Open tasks",
    )


def test_source_hash_stable_and_length() -> None:
    items = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": ""}]
    digest = source_hash(items)
    again = source_hash(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": ""}]
    )
    assert digest == again
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


def test_source_hash_ignores_dict_key_order() -> None:
    assert source_hash([{"b": 2, "a": 1}]) == source_hash([{"a": 1, "b": 2}])


def test_source_hash_is_sensitive() -> None:
    assert source_hash([{"role": "user", "content": "one"}]) != source_hash(
        [{"role": "user", "content": "two"}]
    )
    assert source_hash([{"role": "user", "content": "one"}]) != source_hash(
        [{"role": "assistant", "content": "one"}]
    )


def test_build_request_two_messages_and_sections() -> None:
    request = build_summary_request([_user("hello")], None)
    assert len(request) == 2
    assert request[0]["role"] == "system"
    assert request[1]["role"] == "user"
    instruction = request[0]["content"]
    assert isinstance(instruction, str)
    for section in SECTIONS:
        assert f"## {section}" in instruction
    assert "700" in instruction
    assert "facts" in instruction.lower()


def test_build_request_serializes_tool_calls_and_excludes_reasoning() -> None:
    items = [
        {
            "role": "assistant",
            "content": "Let me write it",
            "reasoning_content": "secret internal chain",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        _tool("ok", "c1"),
    ]
    request = build_summary_request(items, None)
    content = request[1]["content"]
    assert isinstance(content, str)
    assert "[assistant] Let me write it" in content
    assert '[assistant → write_file] {"path": "a.py"}' in content
    assert "[tool] ok" in content
    assert "secret internal chain" not in content


def test_build_request_previous_goes_first() -> None:
    previous = SummaryCheckpoint(
        text="Old facts about earlier work", covers_until=0, source_hash="abc"
    )
    request = build_summary_request([_user("hello")], previous)
    content = request[1]["content"]
    assert content.startswith("Previous summary:\nOld facts about earlier work")
    assert "[user] hello" in content


def test_build_request_truncates_middle() -> None:
    items = [_user("x" * 1000)]
    request = build_summary_request(items, None, max_input_chars=200)
    content = request[1]["content"]
    assert isinstance(content, str)
    assert re.search(r"\.\.\.\[\d+ chars omitted\]\.\.\.", content)
    assert content.startswith("[user] x")
    assert content.endswith("x")
    assert len(content) <= 200
    assert len(content) > 100


def test_normalize_adds_missing_sections_in_order() -> None:
    result = normalize_summary("## Goal\nbuild a tool")
    for section in SECTIONS:
        assert f"## {section}" in result
    assert "- (none)" in result
    positions = [result.index(f"## {section}") for section in SECTIONS]
    assert positions == sorted(positions)


def test_normalize_does_not_duplicate_existing_sections() -> None:
    text = "\n".join(f"## {section}\nnotes" for section in SECTIONS)
    result = normalize_summary(text)
    for section in SECTIONS:
        assert result.count(f"## {section}") == 1
    assert "- (none)" not in result


def test_normalize_truncates_to_limit() -> None:
    result = normalize_summary("## Goal\n" + "y" * 10_000)
    assert len(result) <= 6000


def test_render_summary_item() -> None:
    checkpoint = SummaryCheckpoint(text="summary text", covers_until=7, source_hash="hash")
    item = render_summary_item(checkpoint)
    assert item == {"role": "user", "content": SUMMARY_LABEL + "\n\nsummary text"}


def test_render_summary_item_is_user_role() -> None:
    checkpoint = SummaryCheckpoint(text="data", covers_until=1, source_hash="h")
    assert render_summary_item(checkpoint)["role"] == "user"


def test_deterministic_summary() -> None:
    items: list[dict[str, object]] = [
        _user("Refactor the parser for speed."),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path": "src/parser.py"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": '{"path": "src/parser.py"}'},
                },
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command": "pytest -q"}'},
                },
                {
                    "id": "c4",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{broken json"},
                },
            ],
        },
        _tool('{"success": true, "exit_code": 0}', "c3"),
        _user("Now review the tests and continue."),
    ]
    result = deterministic_summary(items)
    assert result.startswith("## Goal\nRefactor the parser for speed.")
    assert result.count("- src/parser.py") == 1
    assert "## Check results" in result
    assert "pytest -q (exit 0)" in result
    assert "{broken" not in result
    assert result.endswith("## Open tasks\nNow review the tests and continue.")
    assert result.count("## Constraints") == 1
    assert "- (none)" in result
    assert len(result) <= 4000


def test_deterministic_summary_skips_missing_exit_code() -> None:
    items: list[dict[str, object]] = [
        _user("run the suite"),
        _assistant("shell", '{"command": "make check"}', "c1"),
        _tool('{"success": true, "output": "all good"}', "c1"),
    ]
    result = deterministic_summary(items)
    assert "make check" not in result
    assert "## Check results\n- (none)" in result


def test_deterministic_summary_placeholder_without_users() -> None:
    result = deterministic_summary([])
    assert result.startswith("## Goal\n- (none)")
    assert result.endswith("## Open tasks\n- (none)")
    assert result.count("- (none)") == len(SECTIONS)


def test_summary_versions_are_valid_json_roundtrip() -> None:
    checkpoint = SummaryCheckpoint(text="x" * 100, covers_until=3, source_hash="h", version=1)
    data = json.loads(json.dumps(render_summary_item(checkpoint)))
    assert data["role"] == "user"
    assert SUMMARY_LABEL in data["content"]


def test_normalize_limit_keeps_added_sections() -> None:
    result = normalize_summary("## Goal\n" + "x" * 10_000)
    assert len(result) <= 6000
    for section in SECTIONS:
        assert f"## {section}" in result

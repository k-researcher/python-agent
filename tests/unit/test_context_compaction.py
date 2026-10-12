import hashlib
import json

from agent.context_compaction import compact, fold_write_arguments, mask_tool_result


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tool_item(content: object, tool_call_id: str = "call_1") -> dict[str, object]:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


def assistant_item(name: str, arguments: str, call_id: str = "call_1") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def test_mask_json_success() -> None:
    content = json.dumps(
        {"success": True, "output": {"path": "/tmp/f.txt", "bytes_written": 3}},
        ensure_ascii=False,
    )
    result = mask_tool_result(tool_item(content, "call_9"))
    masked = json.loads(result["content"])
    assert masked["masked"] is True
    assert masked["success"] is True
    assert masked["error"] is None
    assert masked["path"] == "/tmp/f.txt"
    assert "path" in masked["excerpt"]
    assert masked["original_chars"] == len(content)
    assert masked["sha256"] == _sha(content)
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_9"


def test_mask_json_error() -> None:
    content = json.dumps(
        {"success": False, "error": "boom: write failed", "output": None}, ensure_ascii=False
    )
    masked = json.loads(mask_tool_result(tool_item(content))["content"])
    assert masked["success"] is False
    assert masked["error"] == "boom: write failed"
    assert masked["excerpt"] == "boom: write failed"


def test_mask_non_json() -> None:
    text = "plain text output"
    masked = json.loads(mask_tool_result(tool_item(text))["content"])
    assert masked["success"] is None
    assert masked["error"] is None
    assert masked["path"] is None
    assert masked["excerpt"] == "plain text output"
    assert masked["original_chars"] == len(text)
    assert masked["sha256"] == _sha(text)


def test_mask_long_output_is_valid_json() -> None:
    content = json.dumps({"success": True, "output": {"data": "y" * 50_000}}, ensure_ascii=False)
    masked = json.loads(mask_tool_result(tool_item(content), excerpt_chars=50)["content"])
    assert masked["original_chars"] == len(content)
    assert masked["sha256"] == _sha(content)
    assert len(masked["excerpt"]) <= 50


def test_mask_does_not_mutate_source() -> None:
    original = {"role": "tool", "content": "x" * 5000, "tool_call_id": "call_7"}
    before = dict(original)
    mask_tool_result(original, excerpt_chars=10)
    assert original == before


def test_mask_keeps_other_roles_unchanged() -> None:
    item = {"role": "user", "content": "hello"}
    result = mask_tool_result(item)
    assert result == item
    assert result is not item


def test_fold_write_file_long_content() -> None:
    before = "a" * 1000
    item = assistant_item("write_file", json.dumps({"path": "a.py", "content": before}))
    result = fold_write_arguments(item)
    call = result["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "write_file"
    args = json.loads(call["function"]["arguments"])
    assert args["content"] == f"[folded: 1000 chars, sha256 {_sha(before)}]"
    assert args["path"] == "a.py"


def test_fold_edit_file_long_fields() -> None:
    old = "z" * 500
    new = "q" * 600
    arguments = json.dumps(
        {"path": "x.txt", "old_text": old, "new_text": new, "replace_all": False}
    )
    result = fold_write_arguments(assistant_item("edit_file", arguments))
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args["old_text"] == f"[folded: 500 chars, sha256 {_sha(old)}]"
    assert args["new_text"] == f"[folded: 600 chars, sha256 {_sha(new)}]"
    assert args["path"] == "x.txt"
    assert args["replace_all"] is False


def test_fold_keeps_short_values() -> None:
    arguments = json.dumps({"path": "a.py", "content": "short"})
    result = fold_write_arguments(assistant_item("write_file", arguments))
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args["content"] == "short"
    assert args["path"] == "a.py"


def test_fold_invalid_arguments_unchanged() -> None:
    item = assistant_item("write_file", "{not json")
    result = fold_write_arguments(item)
    assert result["tool_calls"][0]["function"]["arguments"] == "{not json"


def test_fold_other_tools_unchanged() -> None:
    arguments = json.dumps({"cmd": "ls", "long_data": "x" * 900})
    result = fold_write_arguments(assistant_item("shell", arguments))
    assert result["tool_calls"][0]["function"]["arguments"] == arguments


def test_fold_other_roles_unchanged() -> None:
    item = {"role": "user", "content": "hello"}
    result = fold_write_arguments(item)
    assert result == item
    assert result is not item


def test_compact_preserves_order_length_and_ids() -> None:
    items: list[dict[str, object]] = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
    ]
    for i in range(4):
        arguments = json.dumps({"path": f"f{i}.py", "content": "q" * 700})
        items.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": arguments},
                    }
                ],
            }
        )
        items.append({"role": "tool", "content": f"res {i}", "tool_call_id": f"call_{i}"})
    items.append({"role": "user", "content": "last"})
    before = [dict(item) for item in items]

    result = compact(items, keep_last=2, stage=2)

    assert len(result) == len(items)
    expected_roles = [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
    ]
    assert [item["role"] for item in result] == expected_roles
    assert result[0]["content"] == "rules"
    assert result[1]["content"] == "first"
    assert result[9]["content"] == "res 3"
    assert result[10]["content"] == "last"
    foldable = result[2]
    assert "folded" in json.loads(foldable["tool_calls"][0]["function"]["arguments"])["content"]
    assert json.loads(result[3]["content"])["masked"] is True
    for i in range(4):
        call_id = result[2 + 2 * i]["tool_calls"][0]["id"]
        assert result[3 + 2 * i]["tool_call_id"] == call_id
    assert items == before


def test_compact_stage_controls_folding() -> None:
    arguments = json.dumps({"path": "a.py", "content": "x" * 600})
    items = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
        {"role": "user", "content": "last"},
    ]
    stage1 = compact(items, keep_last=1, stage=1)
    args1 = json.loads(stage1[2]["tool_calls"][0]["function"]["arguments"])
    assert args1["content"] == "x" * 600
    assert json.loads(stage1[3]["content"])["masked"] is True

    stage2 = compact(items, keep_last=1, stage=2)
    args2 = json.loads(stage2[2]["tool_calls"][0]["function"]["arguments"])
    assert args2["content"] == f"[folded: 600 chars, sha256 {_sha('x' * 600)}]"
    assert json.loads(stage2[3]["content"])["masked"] is True

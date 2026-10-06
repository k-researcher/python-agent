import pytest

from agent.context_blocks import Block, flatten, protected_blocks, split_blocks


def item(role: str, content: str = "", **extra: object) -> dict[str, object]:
    result: dict[str, object] = {"role": role, "content": content}
    result.update(extra)
    return result


def calls(*ids: str, content: str = "") -> dict[str, object]:
    return item(
        "assistant",
        content,
        tool_calls=[
            {"id": call_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
            for call_id in ids
        ],
    )


def reply(call_id: str, content: str = "ok") -> dict[str, object]:
    return item("tool", content, tool_call_id=call_id)


_EXAMPLES: list[list[dict[str, object]]] = [
    [],
    [item("system", "rules"), item("user", "hello")],
    [calls("c1", "c2"), reply("c1"), reply("c2"), item("assistant", "done")],
    [item("system", "rules"), item("user", "task"), calls("c1"), reply("c1")],
    [reply("c9"), item("user", "hi"), reply("c8")],
    [
        item("system", "rules"),
        item("user", "first"),
        calls("c1", "c2"),
        reply("c1"),
        calls("c3"),
        reply("c3"),
        item("assistant", "done"),
        item("user", "second"),
    ],
]


def test_empty_input() -> None:
    assert split_blocks([]) == []
    assert protected_blocks([], []) == set()
    assert flatten([], [], []) == []


def test_block_is_frozen() -> None:
    block = Block(0, 1, "user", True)
    assert (block.start, block.end, block.kind, block.complete) == (0, 1, "user", True)


def test_simple_dialog() -> None:
    items = [
        item("system", "rules"),
        item("user", "hello"),
        item("assistant", "hi there"),
        item("user", "again"),
    ]
    blocks = split_blocks(items)
    assert [(b.start, b.end, b.kind, b.complete) for b in blocks] == [
        (0, 1, "system", True),
        (1, 2, "user", True),
        (2, 3, "assistant", True),
        (3, 4, "user", True),
    ]


def test_exchange_with_two_calls() -> None:
    items = [calls("c1", "c2"), reply("c1"), reply("c2"), item("assistant", "done")]
    blocks = split_blocks(items)
    assert [(b.start, b.end, b.kind, b.complete) for b in blocks] == [
        (0, 3, "exchange", True),
        (3, 4, "assistant", True),
    ]


def test_incomplete_exchange_missing_reply() -> None:
    items = [calls("c1", "c2"), reply("c1")]
    blocks = split_blocks(items)
    assert [(b.start, b.end, b.kind, b.complete) for b in blocks] == [(0, 2, "exchange", False)]


def test_orphan_tool() -> None:
    items = [reply("c9"), item("user", "hi"), item("assistant", "ok"), reply("c8")]
    blocks = split_blocks(items)
    assert [(b.start, b.end, b.kind, b.complete) for b in blocks] == [
        (0, 1, "orphan_tool", True),
        (1, 2, "user", True),
        (2, 3, "assistant", True),
        (3, 4, "orphan_tool", True),
    ]


def test_foreign_tool_becomes_orphan() -> None:
    items = [calls("c1"), reply("c1"), reply("c_other"), item("user", "hi")]
    blocks = split_blocks(items)
    assert [(b.start, b.end, b.kind, b.complete) for b in blocks] == [
        (0, 2, "exchange", True),
        (2, 3, "orphan_tool", True),
        (3, 4, "user", True),
    ]


@pytest.mark.parametrize("items", _EXAMPLES)
def test_blocks_cover_all_items(items: list[dict[str, object]]) -> None:
    blocks = split_blocks(items)
    if not items:
        assert blocks == []
        return
    assert blocks[0].start == 0
    assert blocks[-1].end == len(items)
    for previous, following in zip(blocks, blocks[1:], strict=False):
        assert previous.end == following.start


def test_protected_keep_completed_variants() -> None:
    items = [
        item("system", "rules"),
        item("user", "task"),
        calls("c1"),
        reply("c1"),
        calls("c2"),
        reply("c2"),
        item("assistant", "ok"),
        item("user", "again"),
        calls("c3"),
        reply("c3"),
    ]
    blocks = split_blocks(items)
    assert protected_blocks(items, blocks, keep_completed=0) == {0, 1, 5}
    assert protected_blocks(items, blocks, keep_completed=2) == {0, 1, 5, 6, 4}
    assert protected_blocks(items, blocks, keep_completed=5) == {0, 1, 2, 3, 4, 5, 6}


def test_protected_incomplete_exchange() -> None:
    items = [
        item("system", "rules"),
        item("user", "task"),
        calls("c1", "c2"),
        reply("c1"),
    ]
    blocks = split_blocks(items)
    assert protected_blocks(items, blocks) == {0, 1, 2}


def test_protected_late_system_is_not_included() -> None:
    items = [
        item("system", "rules"),
        item("user", "task"),
        item("system", "late"),
        item("user", "more"),
    ]
    blocks = split_blocks(items)
    assert protected_blocks(items, blocks) == {0, 1, 3}


def test_protected_top_up_from_history() -> None:
    items = [
        item("system", "rules"),
        item("user", "task"),
        calls("c1"),
        reply("c1"),
        calls("c2"),
        reply("c2"),
        item("user", "again"),
        calls("c3"),
        reply("c3"),
    ]
    blocks = split_blocks(items)
    assert protected_blocks(items, blocks, keep_completed=2) == {0, 1, 3, 4, 5}


def test_flatten_preserves_order_and_copies() -> None:
    items = [
        item("system", "rules"),
        item("user", "task"),
        calls("c1"),
        reply("c1"),
        item("assistant", "ok"),
        item("user", "last"),
    ]
    blocks = split_blocks(items)
    before = [dict(entry) for entry in items]

    flat = flatten(items, blocks, [4, 2, 0])

    assert [entry["role"] for entry in flat] == ["system", "assistant", "tool", "user"]
    assert [entry["content"] for entry in flat] == ["rules", "", "ok", "last"]
    assert flat[0] is not items[0]
    assert flat[3] is not items[5]
    assert items == before


def test_flatten_accepts_set() -> None:
    items = [item("system", "rules"), item("user", "hi")]
    blocks = split_blocks(items)
    flat = flatten(items, blocks, {1, 0})
    assert [entry["role"] for entry in flat] == ["system", "user"]

from __future__ import annotations

from agent.context_projection import (
    HistorySnapshot,
    SourceMessage,
    build_snapshot,
    chain_hash,
    project,
)
from agent.context_summary import (
    SUMMARY_VERSION,
    SummaryCheckpoint,
    render_summary_item,
    source_hash,
)

VALID_HASH = "a" * 64


def _msg(
    id: int,
    role: str,
    content: str | None = None,
    *,
    kind: str = "normal",
    skipped: bool = False,
    tool_calls: list[dict[str, object]] | None = None,
    tool_call_id: str | None = None,
    reasoning_content: str | None = None,
) -> SourceMessage:
    return SourceMessage(
        id=id,
        role=role,
        kind=kind,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        reasoning_content=reasoning_content,
        skipped=skipped,
    )


def _summary(
    id: int,
    covers_until: int,
    *,
    content: str = "summary text",
    source_hash: str = VALID_HASH,
    version: int = SUMMARY_VERSION,
    skipped: bool = False,
) -> SourceMessage:
    return SourceMessage(
        id=id,
        role="user",
        kind="summary",
        content=content,
        covers_until=covers_until,
        source_hash=source_hash,
        summary_version=version,
        skipped=skipped,
    )


class _FakeRow:
    def __init__(self, **kwargs: object) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)


# --------------------------------------------------------------------------- #
# SourceMessage.from_row
# --------------------------------------------------------------------------- #


def test_from_row_copies_all_attributes() -> None:
    row = _FakeRow(
        id=7,
        role="assistant",
        kind="normal",
        content="hi",
        tool_calls=[{"id": "call_1"}],
        tool_call_id=None,
        reasoning_content="thinking",
        skipped=True,
        covers_until=None,
        source_hash=None,
        summary_version=None,
    )
    parsed = SourceMessage.from_row(row)
    assert parsed == SourceMessage(
        id=7,
        role="assistant",
        kind="normal",
        content="hi",
        tool_calls=[{"id": "call_1"}],
        reasoning_content="thinking",
        skipped=True,
    )


def test_from_row_missing_attributes_use_defaults() -> None:
    partial = _FakeRow(id=3, role="user", kind="normal", content="task")
    parsed = SourceMessage.from_row(partial)
    assert parsed == SourceMessage(id=3, role="user", kind="normal", content="task")
    assert parsed.tool_calls is None
    assert parsed.tool_call_id is None
    assert parsed.reasoning_content is None
    assert parsed.skipped is False
    assert parsed.covers_until is None
    assert parsed.source_hash is None
    assert parsed.summary_version is None


def test_from_row_worst_case_only_id() -> None:
    parsed = SourceMessage.from_row(_FakeRow(id=9))
    assert parsed.id == 9
    assert parsed.role == ""
    assert parsed.kind == "normal"
    assert parsed.content is None
    assert parsed.skipped is False


# --------------------------------------------------------------------------- #
# build_snapshot: message visibility and order
# --------------------------------------------------------------------------- #


def test_messages_exclude_skipped_and_interrupted_in_order() -> None:
    rows = [
        _msg(1, "system", "rules"),
        _msg(2, "user", "task", skipped=True),
        _msg(3, "user", "kept", kind="interrupted"),
        _msg(4, "user", "also kept"),
        _msg(5, "assistant", "", kind="interrupted", skipped=True),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is None
    assert [message.id for message in snapshot.messages] == [1, 4]


def test_empty_rows() -> None:
    snapshot = build_snapshot([])
    assert snapshot.messages == ()
    assert snapshot.checkpoint is None
    assert snapshot.checkpoint_message_id is None


def test_single_message() -> None:
    snapshot = build_snapshot([_msg(1, "user", "only")])
    assert [message.id for message in snapshot.messages] == [1]


def test_summary_rows_never_become_messages() -> None:
    rows = [
        _msg(1, "user", "task"),
        _summary(id=2, covers_until=1),
        _msg(3, "user", "follow-up"),
    ]
    snapshot = build_snapshot(rows)
    assert all(message.kind != "summary" for message in snapshot.messages)
    assert [message.id for message in snapshot.messages] == [1, 3]


# --------------------------------------------------------------------------- #
# build_snapshot: checkpoint selection (bad DB data)
# --------------------------------------------------------------------------- #


def test_checkpoint_selects_greatest_covers_until_then_greatest_id() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "a"),
        _msg(3, "user", "b"),
        _summary(id=10, covers_until=2, content="older"),
        _summary(id=11, covers_until=3, content="newer"),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.text == "newer"
    assert snapshot.checkpoint.covers_until == 3
    assert snapshot.checkpoint_message_id == 11


def test_checkpoint_tie_breaks_by_greatest_id() -> None:
    rows = [
        _msg(1, "user", "task"),
        _summary(id=10, covers_until=1, content="first"),
        _summary(id=12, covers_until=1, content="second"),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.text == "second"
    assert snapshot.checkpoint_message_id == 12


def test_checkpoint_rejects_wrong_version() -> None:
    rows = [_msg(1, "user", "task"), _summary(id=2, covers_until=1, version=SUMMARY_VERSION + 1)]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is None
    assert snapshot.checkpoint_message_id is None
    assert [message.id for message in snapshot.messages] == [1]


def test_checkpoint_rejects_empty_and_whitespace_content() -> None:
    for stale_content in ("", "   ", "\n\t "):
        snapshot = build_snapshot([_msg(1, "user", "task"), _summary(2, 1, content=stale_content)])
        assert snapshot.checkpoint is None, repr(stale_content)


def test_checkpoint_rejects_bad_source_hash() -> None:
    bad_hashes: list[str] = [
        "",  # empty
        "a" * 63,  # too short
        "a" * 65,  # too long
        "z" * 64,  # non-hex character
        "A" * 32 + "g" * 32,  # g is not a hex digit
    ]
    for bad_hash in bad_hashes:
        snapshot = build_snapshot([_msg(1, "user", "task"), _summary(2, 1, source_hash=bad_hash)])
        assert snapshot.checkpoint is None, bad_hash


def test_checkpoint_accepts_uppercase_hex_hash() -> None:
    rows = [_msg(1, "user", "task"), _summary(2, 1, source_hash="A" * 64)]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.source_hash == "A" * 64


def test_checkpoint_rejects_coverage_forward() -> None:
    rows = [_msg(1, "user", "task"), _summary(id=2, covers_until=2)]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is None


def test_checkpoint_rejects_coverage_on_missing_and_summary_rows() -> None:
    missing = build_snapshot([_msg(1, "user", "task"), _summary(id=5, covers_until=99)])
    assert missing.checkpoint is None
    # id=3 covers a summary row (id=2) and must be rejected even when it is the
    # only candidate; id=2 is stubbed as ineligible so nothing else can win.
    summary_row_target = build_snapshot(
        [
            _msg(1, "user", "task"),
            _summary(id=2, covers_until=1, version=SUMMARY_VERSION + 1),
            _summary(id=3, covers_until=2),
        ]
    )
    assert summary_row_target.checkpoint is None


def test_checkpoint_stores_raw_content_not_stripped() -> None:
    raw = "  leading and trailing  "
    snapshot = build_snapshot([_msg(1, "user", "task"), _summary(2, 1, content=raw)])
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.text == raw


def test_skipped_summary_is_not_selected() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "a"),
        _msg(3, "user", "follow-up"),
        _summary(id=4, covers_until=3, skipped=True),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is None
    assert snapshot.checkpoint_message_id is None
    assert [message.id for message in snapshot.messages] == [1, 2, 3]


def test_summary_covers_skipped_row_is_not_selected() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "a", skipped=True),
        _msg(3, "user", "follow-up"),
        _summary(id=4, covers_until=2),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is None
    assert snapshot.checkpoint_message_id is None
    assert [message.id for message in snapshot.messages] == [1, 3]


def test_summary_covers_skipped_row_falls_to_valid() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "a", skipped=True),
        _msg(3, "user", "follow-up"),
        _summary(id=4, covers_until=1, content="small valid"),
        _summary(id=5, covers_until=2, content="covers skipped"),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.text == "small valid"
    assert snapshot.checkpoint_message_id == 4


def test_best_summary_skipped_falls_to_second() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "a"),
        _msg(3, "user", "follow-up"),
        _summary(id=4, covers_until=1, content="small valid"),
        _summary(id=5, covers_until=3, content="hidden large", skipped=True),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.text == "small valid"
    assert snapshot.checkpoint_message_id == 4
    assert [message.id for message in snapshot.messages] == [1, 2, 3]


def test_valid_summary_still_selected_when_hidden_present() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "a"),
        _summary(id=3, covers_until=2, content="visible summary"),
        _summary(id=4, covers_until=2, content="hidden same-coverage", skipped=True),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.text == "visible summary"
    assert snapshot.checkpoint_message_id == 3
    assert [message.id for message in snapshot.messages] == [1]


# --------------------------------------------------------------------------- #
# build_snapshot: guarded message selection
# --------------------------------------------------------------------------- #


def test_checkpoint_guard_keeps_leading_system_first_user_and_tail() -> None:
    rows = [
        _msg(1, "system", "rules"),
        _msg(2, "user", "task"),
        _msg(3, "assistant", "covered work"),
        _msg(4, "assistant", "more covered work"),
        _msg(5, "user", "follow-up"),
        _summary(id=6, covers_until=3),
    ]
    snapshot = build_snapshot(rows)
    assert [message.id for message in snapshot.messages] == [1, 2, 4, 5]


def test_checkpoint_guard_removes_duplicates() -> None:
    rows = [
        _msg(1, "system", "s1"),
        _msg(2, "system", "s2"),
        _msg(3, "user", "task"),
        _summary(id=4, covers_until=1),
    ]
    snapshot = build_snapshot(rows)
    # id=2 appears as a leading system row and again in the covered tail.
    assert [message.id for message in snapshot.messages] == [1, 2, 3]
    assert len(snapshot.messages) == len({message.id for message in snapshot.messages})


def test_checkpoint_guard_first_user_kept_even_when_covered() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "covered"),
        _msg(3, "user", "follow-up"),
        _summary(id=4, covers_until=2),
    ]
    snapshot = build_snapshot(rows)
    assert [message.id for message in snapshot.messages] == [1, 3]


def test_checkpoint_guard_covers_until_boundary_is_inclusive() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "boundary"),
        _msg(3, "user", "after"),
        _summary(id=4, covers_until=2),
    ]
    snapshot = build_snapshot(rows)
    assert [message.id for message in snapshot.messages] == [1, 3]


def test_checkpoint_guard_preserves_explicit_source_order() -> None:
    rows = [
        _msg(1, "system", "rules"),
        _msg(2, "user", "task"),
        _msg(3, "tool", "result"),
        _msg(4, "assistant", "reply"),
        _summary(id=5, covers_until=1),
    ]
    snapshot = build_snapshot(rows)
    ids = [message.id for message in snapshot.messages]
    assert ids == sorted(ids)
    assert ids == [1, 2, 3, 4]


def test_checkpoint_guard_without_user_keeps_systems_and_tail() -> None:
    rows = [
        _msg(1, "system", "s1"),
        _msg(2, "system", "s2"),
        _msg(3, "assistant", "covered"),
        _msg(4, "assistant", "tail"),
        _summary(id=5, covers_until=3),
    ]
    snapshot = build_snapshot(rows)
    assert [message.id for message in snapshot.messages] == [1, 2, 4]


def test_no_checkpoint_keeps_all_normal_messages() -> None:
    rows = [
        _msg(1, "system", "rules"),
        _msg(2, "user", "task"),
        _msg(3, "assistant", "reply"),
    ]
    snapshot = build_snapshot(rows)
    assert snapshot.checkpoint is None
    assert [message.id for message in snapshot.messages] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# project: wire items
# --------------------------------------------------------------------------- #


def test_project_plain_wire_items_with_tool_fields() -> None:
    snapshot = HistorySnapshot(
        messages=(
            _msg(1, "system", "rules"),
            _msg(
                2,
                "assistant",
                "",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "shell"}}],
            ),
            _msg(3, "tool", "ok", tool_call_id="call_1"),
        ),
        checkpoint=None,
        checkpoint_message_id=None,
    )
    result = project(snapshot)
    assert result.items == [
        {"role": "system", "content": "rules"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "shell"}}],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
    ]
    assert result.source_ids == [1, 2, 3]
    assert result.first_user_index is None
    assert result.summary_index is None


def test_project_empty_snapshot() -> None:
    snapshot = HistorySnapshot((), None, None)
    result = project(snapshot)
    assert result.items == []
    assert result.source_ids == []
    assert result.first_user_index is None
    assert result.summary_index is None


# --------------------------------------------------------------------------- #
# project: summary insertion
# --------------------------------------------------------------------------- #


def test_project_inserts_summary_after_first_user() -> None:
    checkpoint = SummaryCheckpoint(
        text="compressed history",
        covers_until=1,
        source_hash=VALID_HASH,
    )
    snapshot = HistorySnapshot(
        messages=(_msg(1, "user", "task"), _msg(2, "assistant", "reply"), _msg(3, "user", "q")),
        checkpoint=checkpoint,
        checkpoint_message_id=None,
    )
    result = project(snapshot)
    assert result.items[0] == {"role": "user", "content": "task"}
    assert result.items[1] == render_summary_item(checkpoint)
    assert result.items[2:] == [
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "q"},
    ]
    assert result.source_ids == [1, None, 2, 3]
    assert result.first_user_index == 0
    assert result.summary_index == 1


def test_project_summary_after_leading_systems_when_no_user() -> None:
    checkpoint = SummaryCheckpoint(
        text="data, not instructions", covers_until=1, source_hash=VALID_HASH
    )
    snapshot = HistorySnapshot(
        messages=(_msg(1, "system", "s1"), _msg(2, "system", "s2"), _msg(3, "assistant", "tail")),
        checkpoint=checkpoint,
        checkpoint_message_id=None,
    )
    result = project(snapshot)
    assert result.items == [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
        render_summary_item(checkpoint),
        {"role": "assistant", "content": "tail"},
    ]
    assert result.source_ids == [1, 2, None, 3]
    assert result.first_user_index is None
    assert result.summary_index == 2


def test_project_integration_with_build_snapshot() -> None:
    rows = [
        _msg(1, "user", "task"),
        _msg(2, "assistant", "covered"),
        _msg(3, "user", "follow-up"),
        _summary(id=4, covers_until=2, content="compressed"),
    ]
    snapshot = build_snapshot(rows)
    result = project(snapshot)
    assert [message.id for message in snapshot.messages] == [1, 3]
    assert result.source_ids == [1, None, 3]
    assert result.summary_index == 1
    assert result.first_user_index == 0


# --------------------------------------------------------------------------- #
# project: reasoning_content policy
# --------------------------------------------------------------------------- #


def _reasoning_snapshot() -> HistorySnapshot:
    return HistorySnapshot(
        messages=(
            _msg(1, "user", "q1"),
            _msg(2, "assistant", "a1", reasoning_content="thinking before follow-up"),
            _msg(3, "user", "q2"),
            _msg(4, "assistant", "a2", reasoning_content="current turn thinking"),
        ),
        checkpoint=None,
        checkpoint_message_id=None,
    )


def test_reasoning_none_never_adds_reasoning() -> None:
    result = project(_reasoning_snapshot(), reasoning_history="none")
    assert all("reasoning_content" not in item for item in result.items)


def test_reasoning_all_adds_every_assistant_reasoning() -> None:
    result = project(_reasoning_snapshot(), reasoning_history="all")
    assert result.items[1]["reasoning_content"] == "thinking before follow-up"
    assert result.items[3]["reasoning_content"] == "current turn thinking"


def test_reasoning_current_turn_only_after_last_user() -> None:
    result = project(_reasoning_snapshot(), reasoning_history="current_turn")
    assert "reasoning_content" not in result.items[1]
    assert result.items[3]["reasoning_content"] == "current turn thinking"


# --------------------------------------------------------------------------- #
# chain_hash
# --------------------------------------------------------------------------- #


def test_chain_hash_stable_and_depends_on_previous_and_items() -> None:
    items_a = [{"role": "user", "content": "one"}]
    items_b = [{"role": "user", "content": "two"}]
    previous = SummaryCheckpoint(text="p", covers_until=1, source_hash="b" * 64)

    base = chain_hash(None, items_a)
    assert base == chain_hash(None, items_a)
    assert base == source_hash([{"previous": ""}, *items_a])

    assert chain_hash(previous, items_a) != base
    assert chain_hash(None, items_b) != base
    assert chain_hash(previous, items_a) != chain_hash(previous, items_b)
    expected_chained = source_hash([{"previous": previous.source_hash}, *items_a])
    assert chain_hash(previous, items_a) == expected_chained


def test_chain_hash_empty_items() -> None:
    assert chain_hash(None, []) == source_hash([{"previous": ""}])
    assert chain_hash(None, []) == chain_hash(None, [])


# --------------------------------------------------------------------------- #
# Repeatability / purity (no duplicates across repeated operations)
# --------------------------------------------------------------------------- #


def test_repeated_operations_are_identical_and_do_not_mutate_input() -> None:
    rows = [
        _msg(1, "system", "rules"),
        _msg(2, "user", "task"),
        _msg(3, "assistant", "covered"),
        _msg(4, "user", "follow-up"),
        _summary(id=5, covers_until=3),
    ]
    first = build_snapshot(rows)
    second = build_snapshot(rows)
    assert first.messages == second.messages
    assert first.checkpoint == second.checkpoint

    projected_a = project(first)
    projected_b = project(second)
    assert projected_a.items == projected_b.items
    assert len(projected_a.items) == len(set(id(item) for item in projected_a.items))
    assert projected_a.source_ids == [1, 2, None, 4]
    # Original rows are untouched: still two system-adjacent messages, one summary.
    assert [row.id for row in rows if row.kind == "summary"] == [5]

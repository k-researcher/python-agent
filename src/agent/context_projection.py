"""Project a stored message history into LLM wire items (stage 2.3c).

Builds a plain, ORM-free snapshot of the source rows and renders it as
OpenAI-style chat items. When a summary checkpoint is present, the projection
keeps only the system preamble, the first user task and the rows after the
covered prefix, and re-emits the checkpoint summary right after the task.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent.context_summary import (
    SUMMARY_VERSION,
    SummaryCheckpoint,
    render_summary_item,
    source_hash,
)

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True, slots=True)
class SourceMessage:
    """Plain copy of one Message row; no ORM objects."""

    id: int
    role: str
    kind: str  # "normal" | "summary" | "interrupted"
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    skipped: bool = False
    covers_until: int | None = None
    source_hash: str | None = None
    summary_version: int | None = None

    @classmethod
    def from_row(cls, row: Any) -> SourceMessage:
        """Copy row attributes by name, falling back to defaults when absent."""
        return cls(
            id=getattr(row, "id", 0),
            role=getattr(row, "role", ""),
            kind=getattr(row, "kind", "normal"),
            content=getattr(row, "content", None),
            tool_calls=deepcopy(getattr(row, "tool_calls", None)),
            tool_call_id=getattr(row, "tool_call_id", None),
            reasoning_content=getattr(row, "reasoning_content", None),
            skipped=getattr(row, "skipped", False),
            covers_until=getattr(row, "covers_until", None),
            source_hash=getattr(row, "source_hash", None),
            summary_version=getattr(row, "summary_version", None),
        )


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """Visible source rows in order and the checkpoint guarding them."""

    messages: tuple[SourceMessage, ...]  # visible source rows in order, no summary rows
    checkpoint: SummaryCheckpoint | None
    checkpoint_message_id: int | None


@dataclass(frozen=True, slots=True)
class Projection:
    """Rendered wire items plus source bookkeeping."""

    items: list[dict[str, Any]]  # wire items in order
    source_ids: list[int | None]  # same length; None for the summary item
    first_user_index: int | None  # index in items of the first real user item
    summary_index: int | None  # index in items of the summary item


def _is_valid_hex(source_hash: str) -> bool:
    """Return True when every character is a hex digit and length is 64."""
    return len(source_hash) == 64 and all(char in _HEX_DIGITS for char in source_hash)


def _eligible_checkpoint(row: SourceMessage, active_source_ids: set[int]) -> bool:
    """Return True when a summary row carries a usable checkpoint payload.

    A usable checkpoint is not itself hidden and its ``covers_until`` points to
    an active (kind == \"normal\", skipped == False) source row of the same
    session with a smaller id.
    """
    if row.skipped:
        return False
    if row.summary_version != SUMMARY_VERSION:
        return False
    if not row.content or not row.content.strip():
        return False
    if row.source_hash is None or not _is_valid_hex(row.source_hash):
        return False
    covers_until = row.covers_until
    if covers_until is None:
        return False
    return covers_until in active_source_ids and covers_until < row.id


def build_snapshot(rows: Sequence[SourceMessage]) -> HistorySnapshot:
    """Pick the best checkpoint and the visible message set for the rows.

    ``rows`` must already be in order. Summary rows never become messages; the
    best checkpoint is the eligible one with the greatest ``covers_until``
    (ties resolved by the greatest row id). With a checkpoint, only the leading
    system rows, the first user row and the rows after ``covers_until`` remain.
    """
    candidate = [row for row in rows if row.kind == "normal" and not row.skipped]
    active_source_ids = {row.id for row in candidate}

    best: SourceMessage | None = None
    best_key = (-1, -1)
    for row in rows:
        if row.kind != "summary" or not _eligible_checkpoint(row, active_source_ids):
            continue
        key = (row.covers_until or 0, row.id)
        if key > best_key:
            best_key = key
            best = row

    checkpoint: SummaryCheckpoint | None = None
    checkpoint_message_id: int | None = None
    covers_until = 0
    if best is not None and best.covers_until is not None:
        covers_until = best.covers_until
        checkpoint = SummaryCheckpoint(
            text=best.content or "",
            covers_until=covers_until,
            source_hash=best.source_hash or "",
            version=best.summary_version or SUMMARY_VERSION,
        )
        checkpoint_message_id = best.id

    if checkpoint is None:
        return HistorySnapshot(tuple(candidate), None, None)

    selected: list[SourceMessage] = []
    seen_ids: set[int] = set()

    def add(row: SourceMessage) -> None:
        if row.id not in seen_ids:
            seen_ids.add(row.id)
            selected.append(row)

    cursor = 0
    while cursor < len(candidate) and candidate[cursor].role == "system":
        add(candidate[cursor])
        cursor += 1
    first_user = next((row for row in candidate if row.role == "user"), None)
    if first_user is not None:
        add(first_user)
    for row in candidate:
        if row.id > covers_until:
            add(row)
    return HistorySnapshot(tuple(selected), checkpoint, checkpoint_message_id)


def project(snapshot: HistorySnapshot, *, reasoning_history: str = "none") -> Projection:
    """Render the snapshot as wire items, inserting the summary after the task."""
    messages = snapshot.messages
    last_user = max(
        (index for index, message in enumerate(messages) if message.role == "user"),
        default=-1,
    )
    items: list[dict[str, Any]] = []
    source_ids: list[int | None] = []
    for position, message in enumerate(messages):
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if (
            message.role == "assistant"
            and message.reasoning_content
            and (
                reasoning_history == "all"
                or (reasoning_history == "current_turn" and position > last_user)
            )
        ):
            item["reasoning_content"] = message.reasoning_content
        items.append(item)
        source_ids.append(message.id)

    first_user_index = next(
        (index for index, item in enumerate(items) if item["role"] == "user"), None
    )
    summary_index: int | None = None
    if snapshot.checkpoint is not None:
        if first_user_index is not None:
            insert_at = first_user_index + 1
        else:
            insert_at = 0
            while insert_at < len(items) and items[insert_at]["role"] == "system":
                insert_at += 1
        items.insert(insert_at, render_summary_item(snapshot.checkpoint))
        source_ids.insert(insert_at, None)
        summary_index = insert_at
    return Projection(items, source_ids, first_user_index, summary_index)


def chain_hash(previous: SummaryCheckpoint | None, items: list[dict[str, Any]]) -> str:
    """Return the chained hash of the previous checkpoint and new wire items."""
    payload: list[dict[str, Any]] = [
        {"previous": previous.source_hash if previous is not None else ""}
    ]
    payload.extend(items)
    return source_hash(payload)

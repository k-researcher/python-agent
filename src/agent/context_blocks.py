"""Split LLM context items into atomic blocks and pick protected blocks.

A block is a contiguous run of OpenAI-style chat items that stays together,
such as an assistant tool call plus its tool replies. Protected blocks are
the ones that survive context trims without breaking calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

BlockKind = Literal["system", "user", "assistant", "exchange", "orphan_tool"]

_COMPLETED_KINDS = frozenset[BlockKind](("exchange", "assistant"))


@dataclass(frozen=True, slots=True)
class Block:
    """A contiguous run of items that belongs together."""

    start: int
    end: int
    kind: BlockKind
    complete: bool


def _exchange_end(items: list[dict[str, Any]], start: int) -> tuple[int, bool]:
    """Return the end index and completeness of an exchange block."""
    calls = items[start].get("tool_calls") or []
    ids = {
        call["id"] for call in calls if isinstance(call, dict) and isinstance(call.get("id"), str)
    }
    end = start + 1
    replied: set[str] = set()
    while (
        end < len(items)
        and items[end].get("role") == "tool"
        and items[end].get("tool_call_id") in ids
    ):
        replied.add(items[end]["tool_call_id"])
        end += 1
    return end, ids.issubset(replied)


def split_blocks(items: list[dict[str, Any]]) -> list[Block]:
    """Split items into ordered blocks that cover every item exactly once."""
    blocks: list[Block] = []
    index = 0
    while index < len(items):
        role = items[index].get("role")
        if role == "assistant" and items[index].get("tool_calls"):
            end, complete = _exchange_end(items, index)
            blocks.append(Block(index, end, "exchange", complete))
        elif role == "assistant":
            blocks.append(Block(index, index + 1, "assistant", True))
        elif role == "tool":
            blocks.append(Block(index, index + 1, "orphan_tool", True))
        elif role == "system":
            blocks.append(Block(index, index + 1, "system", True))
        else:
            blocks.append(Block(index, index + 1, "user", True))
        index = blocks[-1].end
    return blocks


def protected_blocks(
    items: list[dict[str, Any]],
    blocks: list[Block],
    *,
    keep_completed: int = 2,
) -> set[int]:
    """Return block indices that must stay when the context is trimmed."""
    protected: set[int] = set()

    first_other = next(
        (index for index, block in enumerate(blocks) if block.kind != "system"),
        len(blocks),
    )
    protected.update(range(first_other))

    user_indices = [index for index, block in enumerate(blocks) if block.kind == "user"]
    if user_indices:
        protected.add(user_indices[0])
        protected.add(user_indices[-1])

    for index, block in enumerate(blocks):
        if block.kind == "exchange" and not block.complete:
            protected.add(index)

    last_user = user_indices[-1] if user_indices else -1
    recent = [
        index
        for index, block in enumerate(blocks)
        if index > last_user and block.complete and block.kind in _COMPLETED_KINDS
    ]
    remaining = 0
    if keep_completed > 0:
        chosen = recent[-keep_completed:]
        protected.update(chosen)
        remaining = keep_completed - len(chosen)
    if remaining > 0:
        history = [
            index
            for index, block in enumerate(blocks)
            if block.complete and block.kind in _COMPLETED_KINDS
        ]
        for index in reversed(history):
            if remaining == 0:
                break
            if index not in protected:
                protected.add(index)
                remaining -= 1
    return protected


def flatten(
    items: list[dict[str, Any]],
    blocks: list[Block],
    indices: Iterable[int],
) -> list[dict[str, Any]]:
    """Return copied items of the chosen blocks in their original order."""
    result: list[dict[str, Any]] = []
    for block_index in sorted(indices):
        block = blocks[block_index]
        result.extend(dict(item) for item in items[block.start : block.end])
    return result

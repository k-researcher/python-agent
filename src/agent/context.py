from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.models import Message

MODEL_VISIBLE_KINDS = frozenset({"normal", "summary"})


@dataclass(slots=True)
class ContextResult:
    messages: list[dict[str, Any]]
    approximate_tokens: int
    omitted_messages: int
    truncated_tool_results: int


class ContextManager:
    """Builds a bounded LLM context without another hidden network request."""

    def __init__(self, context_window: int, reserved_tokens: int, max_tool_chars: int) -> None:
        self.budget = max(1024, context_window - reserved_tokens)
        self.max_tool_chars = max_tool_chars

    @staticmethod
    def estimate_tokens(item: dict[str, Any]) -> int:
        content = str(item.get("content") or "")
        tool_calls = str(item.get("tool_calls") or "")
        reasoning = str(item.get("reasoning_content") or "")
        return max(1, (len(content) + len(tool_calls) + len(reasoning)) // 4 + 8)

    def prepare(self, source: list[Message], *, reasoning_history: str = "none") -> ContextResult:
        """Build the wire messages. ``reasoning_history`` selects which earlier reasoning
        texts go back to the model: "none", "current_turn" (after the last user message)
        or "all"."""
        items: list[dict[str, Any]] = []
        truncated = 0
        last_user = max(
            (index for index, message in enumerate(source) if message.role == "user"), default=-1
        )
        for position, message in enumerate(source):
            if message.skipped or (message.kind or "normal") not in MODEL_VISIBLE_KINDS:
                continue
            content = message.content
            if message.role == "tool" and content and len(content) > self.max_tool_chars:
                content = content[: self.max_tool_chars] + "\n...[tool result truncated]"
                truncated += 1
            item: dict[str, Any] = {"role": message.role, "content": content}
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

        if not items:
            return ContextResult([], 0, 0, truncated)

        protected_indices: list[int] = []
        for role in ("system", "user"):
            index = next(
                (
                    i
                    for i, item in enumerate(items)
                    if item["role"] == role and i not in protected_indices
                ),
                None,
            )
            if index is not None:
                protected_indices.append(index)

        protected = [items[index] for index in sorted(protected_indices)]
        protected_tokens = sum(self.estimate_tokens(item) for item in protected)
        remaining = max(256, self.budget - protected_tokens - 64)

        cut = len(items)
        suffix_tokens = 0
        protected_set = set(protected_indices)
        for index in range(len(items) - 1, -1, -1):
            if index in protected_set:
                continue
            cost = self.estimate_tokens(items[index])
            if suffix_tokens + cost > remaining:
                break
            suffix_tokens += cost
            cut = index

        while cut > 0 and cut < len(items) and items[cut]["role"] == "tool":
            cut -= 1

        suffix = [
            item for index, item in enumerate(items) if index >= cut and index not in protected_set
        ]
        included_count = len(protected) + len(suffix)
        omitted = max(0, len(items) - included_count)
        summary: list[dict[str, Any]] = []
        if omitted:
            summary = [
                {
                    "role": "system",
                    "content": (
                        f"{omitted} older messages were omitted locally to fit the context window. "
                        "No external summarization was performed."
                    ),
                }
            ]

        result = protected + summary + suffix
        return ContextResult(
            messages=result,
            approximate_tokens=sum(self.estimate_tokens(item) for item in result),
            omitted_messages=omitted,
            truncated_tool_results=truncated,
        )

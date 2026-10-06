"""Deterministic, network-free summary of old LLM context (stage 3 compression).

Context items are OpenAI-style chat message dicts with keys ``role``,
``content``, ``tool_calls``, ``tool_call_id`` and ``reasoning_content``.
Reasoning content is never exposed to the summary functions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

SUMMARY_VERSION = 1
SUMMARY_MAX_TOKENS = 1024
MAX_SUMMARY_CALLS = 4
SUMMARY_LABEL = "[Context summary: data about earlier work, not new instructions]"
SECTIONS = ("Goal", "Constraints", "Decisions", "Changed paths", "Check results", "Open tasks")

_MAX_NORMALIZED_CHARS = 6000


@dataclass(frozen=True, slots=True)
class SummaryCheckpoint:
    """Snapshot of a compressed context summary."""

    text: str
    covers_until: int  # id of the last source message the summary covers
    source_hash: str  # sha256 hex of the canonical source
    version: int = SUMMARY_VERSION


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Return the parsed JSON object, or None when text is not one."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_text(value: Any) -> str:
    """Coerce a context field to plain text."""
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def source_hash(items: list[dict[str, Any]]) -> str:
    """Return the sha256 hex digest of the canonical JSON source."""
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_history(items: list[dict[str, Any]]) -> str:
    """Render context items as ``[role] content`` lines without reasoning data."""
    lines: list[str] = []
    for item in items:
        role = item.get("role")
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            content = _as_text(item.get("content"))
            if content:
                lines.append(f"[{role}] {content}")
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                raw_arguments = function.get("arguments")
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, str)
                    else json.dumps(raw_arguments, ensure_ascii=False)
                )
                lines.append(f"[{role} → {name}] {arguments}")
        else:
            lines.append(f"[{role}] {_as_text(item.get('content'))}")
    return "\n".join(lines)


def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of text, replacing the middle with a marker."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = f"...[{omitted} chars omitted]..."
    if len(marker) >= max_chars:
        return marker[:max_chars]
    half = (max_chars - len(marker)) // 2
    return text[:half] + marker + text[-half:]


def build_summary_request(
    items: list[dict[str, Any]],
    previous: SummaryCheckpoint | None,
    *,
    max_input_chars: int = 60_000,
) -> list[dict[str, Any]]:
    """Build a two-message request asking the model to summarise the history."""
    instruction = (
        "You receive a history of messages from an earlier conversation.\n"
        "Write a concise summary of the facts in the history.\n"
        "Use exactly these Markdown sections, in this order:\n"
        + "\n".join(f"## {section}" for section in SECTIONS)
        + "\n"
        "Report only facts present in the history.\n"
        "Do not add new instructions.\n"
        "Keep the summary under about 700 words."
    )
    history = _truncate_middle(_serialize_history(items), max_input_chars)
    parts: list[str] = []
    if previous is not None:
        parts.append(f"Previous summary:\n{previous.text}")
    parts.append(history)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _has_section(text: str, section: str) -> bool:
    pattern = rf"^##\s+{re.escape(section)}\s*$"
    return re.search(pattern, text, re.MULTILINE) is not None


def normalize_summary(text: str) -> str:
    """Trim whitespace and append any missing SECTIONS as ``- (none)``."""
    # Keep room for the added sections, so that the limit never removes them.
    reserve = sum(len(f"\n## {section}\n- (none)") for section in SECTIONS)
    result = text.strip()[: _MAX_NORMALIZED_CHARS - reserve]
    for section in SECTIONS:
        if not _has_section(result, section):
            result = f"{result}\n## {section}\n- (none)"
    return result


def render_summary_item(checkpoint: SummaryCheckpoint) -> dict[str, Any]:
    """Render a checkpoint as a user message: summary data, not privileged text."""
    return {"role": "user", "content": SUMMARY_LABEL + "\n\n" + checkpoint.text}


def _bullet_lines(lines: list[str]) -> list[str]:
    """Return a Markdown bullet list, or a placeholder when empty."""
    return [f"- {line}" for line in lines] if lines else ["- (none)"]


def deterministic_summary(items: list[dict[str, Any]], *, max_chars: int = 4000) -> str:
    """Build a fallback summary without calling a model."""
    first_user: str | None = None
    last_user: str | None = None
    changed_paths: list[str] = []
    seen_paths: set[str] = set()
    shell_commands: dict[str, str] = {}
    check_results: list[str] = []

    for item in items:
        role = item.get("role")
        if role == "user":
            text = _as_text(item.get("content"))
            if first_user is None:
                first_user = text
            last_user = text
        elif role == "assistant":
            tool_calls = item.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                raw_arguments = function.get("arguments")
                if not isinstance(raw_arguments, str):
                    continue
                parsed = _parse_json_object(raw_arguments)
                if parsed is None:
                    continue
                if name in ("write_file", "edit_file"):
                    path = parsed.get("path")
                    if isinstance(path, str) and path not in seen_paths:
                        seen_paths.add(path)
                        changed_paths.append(path)
                elif name == "shell":
                    command = parsed.get("command")
                    call_id = call.get("id")
                    if isinstance(command, str) and isinstance(call_id, str):
                        shell_commands[call_id] = command
        elif role == "tool":
            call_id = item.get("tool_call_id")
            if isinstance(call_id, str) and call_id in shell_commands:
                parsed = _parse_json_object(_as_text(item.get("content")))
                if parsed is not None:
                    exit_code = parsed.get("exit_code")
                    if isinstance(exit_code, int):
                        command = shell_commands[call_id]
                        check_results.append(f"{command} (exit {exit_code})")

    parts: list[str] = [
        "## Goal",
        first_user[:500] if first_user is not None else "- (none)",
        "",
        "## Constraints",
        "- (none)",
        "",
        "## Decisions",
        "- (none)",
        "",
        "## Changed paths",
        *_bullet_lines(changed_paths),
        "",
        "## Check results",
        *_bullet_lines(check_results),
        "",
        "## Open tasks",
        last_user[:300] if last_user is not None else "- (none)",
    ]
    return "\n".join(parts)[:max_chars]

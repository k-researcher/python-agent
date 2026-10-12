"""Non-destructive compaction of old LLM context messages (stages 1 and 2).

Context items are OpenAI-style chat message dicts with keys ``role``,
``content``, ``tool_calls``, ``tool_call_id`` and ``reasoning_content``.
Input dicts are never mutated; every function returns new dicts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_FOLD_KEYS = ("content", "old_text", "new_text")
_TARGET_TOOLS = frozenset({"write_file", "edit_file"})


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Return the parsed JSON object, or None when text is not one."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _excerpt(output: Any, error: str | None, excerpt_chars: int) -> str:
    """Serialise the preferred source and take its leading characters."""
    if error is not None:
        source = str(error)
    elif output is not None:
        source = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    else:
        source = ""
    return source[:excerpt_chars]


def mask_tool_result(item: dict[str, Any], *, excerpt_chars: int = 200) -> dict[str, Any]:
    """Mask a ``tool`` message with a compact JSON summary of its result."""
    if item.get("role") != "tool":
        return dict(item)
    content = item.get("content")
    text = content if isinstance(content, str) else ("" if content is None else str(content))
    parsed = _parse_json_object(text)
    if parsed is not None:
        raw_success: Any = parsed.get("success")
        success: bool | None = raw_success if isinstance(raw_success, bool) else None
        raw_error: Any = parsed.get("error")
        error: str | None = str(raw_error) if raw_error is not None else None
        output: Any = parsed.get("output")
        path: str | None = None
        if isinstance(output, dict):
            raw_path = output.get("path")
            if isinstance(raw_path, str):
                path = raw_path
        excerpt = _excerpt(output, error, excerpt_chars)
    else:
        success = None
        error = None
        path = None
        excerpt = text[:excerpt_chars]
    masked: dict[str, Any] = {
        "masked": True,
        "success": success,
        "error": error,
        "path": path,
        "original_chars": len(text),
        "sha256": _sha256_short(text),
        "excerpt": excerpt,
    }
    result = dict(item)
    result["content"] = json.dumps(masked, ensure_ascii=False, separators=(",", ":"))
    return result


def _fold_arguments(parsed: dict[str, Any], limit_chars: int) -> dict[str, Any]:
    """Return parsed with long content/old_text/new_text folded into summaries."""
    folded = dict(parsed)
    changed = False
    for key in _FOLD_KEYS:
        value = folded.get(key)
        if isinstance(value, str) and len(value) > limit_chars:
            folded[key] = f"[folded: {len(value)} chars, sha256 {_sha256_short(value)}]"
            changed = True
    return folded if changed else parsed


def fold_write_arguments(item: dict[str, Any], *, limit_chars: int = 400) -> dict[str, Any]:
    """Fold long write/edit arguments of assistant tool calls into summaries."""
    if item.get("role") != "assistant":
        return dict(item)
    tool_calls = item.get("tool_calls")
    if not isinstance(tool_calls, list):
        return dict(item)
    folded_calls: list[dict[str, Any]] = []
    changed = False
    for call in tool_calls:
        if not isinstance(call, dict):
            folded_calls.append(call)
            continue
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") not in _TARGET_TOOLS:
            folded_calls.append(call)
            continue
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            folded_calls.append(call)
            continue
        parsed = _parse_json_object(raw_arguments)
        if parsed is None:
            folded_calls.append(call)
            continue
        folded_args = _fold_arguments(parsed, limit_chars)
        if folded_args is parsed:
            folded_calls.append(call)
            continue
        changed = True
        new_function = dict(function)
        new_function["arguments"] = json.dumps(
            folded_args, ensure_ascii=False, separators=(",", ":")
        )
        new_call = dict(call)
        new_call["function"] = new_function
        folded_calls.append(new_call)
    if not changed:
        return dict(item)
    result = dict(item)
    result["tool_calls"] = folded_calls
    return result


def compact(
    items: list[dict[str, Any]], *, keep_last: int = 6, stage: int = 1
) -> list[dict[str, Any]]:
    """Compress older messages while preserving protected boundaries and IDs."""
    if not items:
        return []
    length = len(items)
    protected: set[int] = set()

    def _first_of(role: str) -> None:
        for index, item in enumerate(items):
            if item.get("role") == role:
                protected.add(index)
                return

    _first_of("system")
    _first_of("user")
    for index in range(length - 1, -1, -1):
        if items[index].get("role") == "user":
            protected.add(index)
            break
    protected.update(range(max(0, length - keep_last), length))

    result: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if index in protected:
            result.append(dict(item))
        elif stage >= 2:
            result.append(fold_write_arguments(mask_tool_result(item)))
        elif stage >= 1:
            result.append(mask_tool_result(item))
        else:
            result.append(dict(item))
    return result

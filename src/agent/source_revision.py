"""Deterministic hashing of tool-source definitions and JSON Schemas.

Hashing is strict: only JSON-native values are accepted, JSON object keys must be
strings, dictionary keys and set-valued source fields are sorted, and argv-style
sequences keep their order. Non-JSON values (sets, tuples, unsupported objects),
non-string keys, non-finite floats, and cyclic containers are rejected with
ValueError.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from agent.tool_sources import ToolSource

_SEPARATORS = (",", ":")


def _json_error(message: str) -> ValueError:
    return ValueError(message)


def _canonical(value: Any, _seen: set[int] | None = None) -> Any:
    """Recursively validate and canonicalize a JSON-native value.

    Only ``None``, ``bool``, ``int``, finite ``float``, ``str``, ``list``, and
    ``dict`` with string keys are accepted. Non-string keys, sets, tuples,
    non-finite floats, cyclic containers, and unsupported types raise ValueError.
    Dictionary keys are sorted by their string value; list order is preserved.
    """
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _json_error("Non-finite float is not JSON-serializable")
        return value
    if isinstance(value, (set, frozenset, tuple)):
        raise _json_error(f"Unsupported JSON value of type {type(value).__name__}")
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in _seen:
            raise _json_error("Cycle detected in JSON value")
        _seen.add(identity)
        try:
            if isinstance(value, list):
                return [_canonical(item, _seen) for item in value]
            for key in value:
                if not isinstance(key, str):
                    raise _json_error(f"JSON object key must be a string, got {type(key).__name__}")
            canonical: dict[str, Any] = {}
            for key in sorted(value):
                canonical[key] = _canonical(value[key], _seen)
            return canonical
        finally:
            _seen.discard(identity)
    raise _json_error(f"Unsupported JSON value of type {type(value).__name__}")


def _hash(payload: dict[str, Any]) -> str:
    canonical = _canonical(payload)
    raw = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=_SEPARATORS,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_revision(source: ToolSource) -> str:
    """Return a deterministic SHA256 over every field of a ToolSource.

    All model fields are included via ``model_dump(mode=\"json\")``, so future
    fields are covered automatically. The unordered source fields ``project_ids``
    and ``allowed_modes`` are sorted; command and environment-key argv order is
    preserved. The source object is not mutated.
    """
    payload = source.model_dump(mode="json")
    payload["project_ids"] = sorted(payload["project_ids"])
    payload["allowed_modes"] = sorted(payload["allowed_modes"])
    return _hash(payload)


def schema_hash(schema: dict[str, Any]) -> str:
    """Return a deterministic SHA256 over a strict JSON Schema.

    Only JSON-native values are accepted. Dictionary keys are sorted by their
    string value; list order is preserved. Non-string keys, sets, tuples,
    non-finite numbers, cyclic containers, and unsupported types are rejected.
    """
    if not isinstance(schema, dict):
        raise ValueError("JSON Schema must be an object")
    return _hash(schema)

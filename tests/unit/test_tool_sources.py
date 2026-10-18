"""Test trusted source definitions, strict validation, limits, and redaction."""

import io
import json
import traceback
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.tool_sources import ToolSourceError, load_tool_sources
from agent.tools.base import RiskLevel

_VALID = {"source_id": "server-1", "workspace_id": "Workspace-UUID", "command": ["server"]}


def _config(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_defaults_and_explicit_configuration(tmp_path: Path) -> None:
    assert load_tool_sources(None) == ()
    assert load_tool_sources(_config(tmp_path, {"sources": []})) == ()
    source = load_tool_sources(_config(tmp_path, {"sources": [_VALID]}))[0]
    assert source.enabled is False
    assert source.risk_level is RiskLevel.network_access
    assert source.allowed_modes == frozenset({"dev"})
    assert source.project_ids == frozenset()
    assert source.env_keys == ()
    assert source.command == ("server",)
    assert source.max_result_chars == 20_000
    configured = {
        **_VALID,
        "workspace_id": " Workspace-UUID ",
        "project_ids": ["Project-UUID"],
        "env_keys": ["ACCESS_TOKEN"],
        "enabled": True,
        "risk_level": "read_only",
        "allowed_modes": ["dev", "ask"],
        "max_result_chars": 1,
    }
    source = load_tool_sources(_config(tmp_path, {"sources": [configured]}))[0]
    assert source.workspace_id == " Workspace-UUID "
    assert source.project_ids == frozenset({"Project-UUID"})
    assert source.env_keys == ("ACCESS_TOKEN",)
    assert source.enabled is True
    assert source.risk_level is RiskLevel.read_only
    assert source.allowed_modes == frozenset({"dev", "ask"})
    assert source.max_result_chars == 1
    for field in ("command", "env_keys", "project_ids", "source_id", "enabled"):
        with pytest.raises(ValidationError):
            setattr(source, field, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "Bad-ID"),
        ("source_id", "valid\n"),
        ("source_id", "x" * 33),
        ("workspace_id", ""),
        ("workspace_id", "  "),
        ("workspace_id", "ws\x00"),
        ("workspace_id", 123),
        ("project_ids", [""]),
        ("project_ids", ["\x00"]),
        ("project_ids", "project"),
        ("command", []),
        ("command", [""]),
        ("command", ["  "]),
        ("command", ["bad\x00"]),
        ("command", "server"),
        ("command", [False]),
        ("env_keys", ["TOKEN=value"]),
        ("env_keys", ["1TOKEN"]),
        ("env_keys", ["TOKEN\n"]),
        ("env_keys", {"TOKEN": "value"}),
        ("enabled", 1),
        ("enabled", "true"),
        ("risk_level", "safe"),
        ("allowed_modes", ["unknown"]),
        ("allowed_modes", "dev"),
        ("max_result_chars", True),
        ("max_result_chars", 1.0),
        ("max_result_chars", "1"),
        ("max_result_chars", 0),
        ("max_result_chars", 1_000_001),
        ("unknown", "value"),
    ],
)
def test_invalid_source_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ToolSourceError):
        load_tool_sources(_config(tmp_path, {"sources": [{**_VALID, field: value}]}))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "text",
        {},
        {"sources": {}},
        {"sources": None},
        {"sources": [], "extra": 1},
        {"sources": [None]},
        {"sources": ["source"]},
        {"sources": [1]},
    ],
)
def test_invalid_document_is_rejected(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ToolSourceError):
        load_tool_sources(_config(tmp_path, payload))


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"\xff",
        b'{"sources":[],"sources":[]}',
        b'{"sources":[{"source_id":"server","workspace_id":"ws","command":["server"],'
        b'"enabled":true,"enabled":false}]}',
        b"[" * 2000,
    ],
)
def test_invalid_encoding_json_and_duplicate_keys(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "sources.json"
    path.write_bytes(raw)
    with pytest.raises(ToolSourceError):
        load_tool_sources(path)


def test_scope_identity_and_source_count_limit(tmp_path: Path) -> None:
    other = {**_VALID, "workspace_id": "another"}
    assert len(load_tool_sources(_config(tmp_path, {"sources": [_VALID, other]}))) == 2
    with pytest.raises(ToolSourceError):
        load_tool_sources(_config(tmp_path, {"sources": [_VALID, _VALID]}))
    sources = [{**_VALID, "source_id": f"source-{index}"} for index in range(101)]
    assert len(load_tool_sources(_config(tmp_path, {"sources": sources[:100]}))) == 100
    with pytest.raises(ToolSourceError):
        load_tool_sources(_config(tmp_path, {"sources": sources}))


def test_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[int] = []

    class BoundedFile(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            assert 0 < size <= (1 << 20) + 1
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *args: BoundedFile(b" " * ((1 << 20) + 2)))
    with pytest.raises(ToolSourceError):
        load_tool_sources(tmp_path / "huge.json")
    assert requested == [(1 << 20) + 1]


@pytest.mark.parametrize("missing", [False, True])
def test_error_and_traceback_do_not_disclose_input(tmp_path: Path, missing: bool) -> None:
    secret = "private-canary-12345"
    path = tmp_path / secret
    if not missing:
        path.write_text(json.dumps({"sources": [{**_VALID, "unknown": secret}]}))
    with pytest.raises(ToolSourceError) as caught:
        load_tool_sources(path)
    assert secret not in str(caught.value)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__ is True

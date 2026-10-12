from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.safe_fs import SafeProjectFS
from agent.security import PathGuard
from agent.tools import ToolRegistry
from agent.tools.base import ToolContext, ToolError
from agent.tools.builtin import (
    MAX_TEXT_BYTES,
    GlobTool,
    MultiEditTool,
    ReadManyTool,
)


def context(root: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=root)


async def test_glob_finds_files_and_respects_exclusions_and_limit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src/a.py").write_text("x")
    (project / "src/b.py").write_text("x")
    (project / "src/.venv").mkdir()
    (project / "src/.venv/ignored.py").write_text("x")
    (project / "src/__pycache__").mkdir()
    (project / "src/__pycache__/cache.py").write_text("x")

    full = await GlobTool().execute(context(project), {"pattern": "**/*.py"})
    assert set(full["paths"]) == {"src/a.py", "src/b.py"}
    assert full["truncated"] is False

    limited = await GlobTool().execute(context(project), {"pattern": "**/*.py", "max_results": 1})
    assert limited["paths"] == ["src/a.py"]
    assert limited["truncated"] is True


async def test_glob_truncated_only_when_extra_match_exists(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (project / name).write_text("x")

    fewer = await GlobTool().execute(context(project), {"pattern": "*.py", "max_results": 5})
    assert fewer == {"paths": ["a.py", "b.py", "c.py"], "truncated": False}

    exact = await GlobTool().execute(context(project), {"pattern": "*.py", "max_results": 3})
    assert exact == {"paths": ["a.py", "b.py", "c.py"], "truncated": False}

    more = await GlobTool().execute(context(project), {"pattern": "*.py", "max_results": 2})
    assert len(more["paths"]) == 2
    assert set(more["paths"]) < {"a.py", "b.py", "c.py"}
    assert more["truncated"] is True


async def test_glob_skips_symlink_escaping_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret")
    (project / "link.py").symlink_to(outside)
    (project / "real.py").write_text("x")

    result = await GlobTool().execute(context(project), {"pattern": "*.py"})

    assert result["paths"] == ["real.py"]
    assert result["truncated"] is False


@pytest.mark.parametrize("external_dir", ["node_modules", ".venv", ".git"])
async def test_glob_project_root_inside_external_excluded_dir(
    tmp_path: Path, external_dir: str
) -> None:
    """Parent dirs outside the project must not exclude files inside the root."""
    project = tmp_path / external_dir / "library"
    (project / "src").mkdir(parents=True)
    (project / "src/a.py").write_text("x")
    (project / "src/deps").mkdir()
    (project / "src/deps/node_modules").mkdir()
    (project / "src/deps/node_modules/ignored.py").write_text("x")

    result = await GlobTool().execute(context(project), {"pattern": "**/*.py"})

    assert result["paths"] == ["src/a.py"]
    assert result["truncated"] is False


async def test_glob_takes_lexicographically_first_with_reverse_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lexicographic-first contract must not depend on filesystem walk order."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (project / name).write_text("x")
    real_glob = Path.glob

    def reversed_glob(self: Path, pattern: str):
        yield from reversed(list(real_glob(self, pattern)))

    monkeypatch.setattr(Path, "glob", reversed_glob)

    result = await GlobTool().execute(context(project), {"pattern": "*.py", "max_results": 1})

    assert result["paths"] == ["a.py"]
    assert result["truncated"] is True


async def test_read_many_reads_files_and_reports_single_error(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")

    result = await ReadManyTool().execute(
        context(tmp_path),
        {"paths": ["a.txt", "missing.txt", "b.txt"], "max_bytes_per_file": 1000},
    )

    assert result["files"][0] == {"path": "a.txt", "content": "hello", "truncated": False}
    assert result["files"][1]["path"] == "missing.txt"
    assert "error" in result["files"][1]
    assert result["files"][2] == {"path": "b.txt", "content": "world", "truncated": False}


async def test_read_many_truncates_large_file(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("a" * 5000)
    (tmp_path / "small.txt").write_text("ok")

    result = await ReadManyTool().execute(
        context(tmp_path),
        {"paths": ["big.txt", "small.txt"], "max_bytes_per_file": 1000},
    )

    assert result["files"][0] == {"path": "big.txt", "content": "a" * 1000, "truncated": True}
    assert result["files"][1] == {"path": "small.txt", "content": "ok", "truncated": False}


async def test_read_many_truncates_multibyte_at_char_boundary(tmp_path: Path) -> None:
    (tmp_path / "emoji.txt").write_text("я🙂x", encoding="utf-8")

    result = await ReadManyTool().execute(
        context(tmp_path),
        {"paths": ["emoji.txt"], "max_bytes_per_file": 5},
    )

    assert result["files"][0] == {"path": "emoji.txt", "content": "я", "truncated": True}


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("big", b"a" * (MAX_TEXT_BYTES + 1), "larger than"),
        ("corrupt", b"a" * 20 + b"\xff", "UTF-8"),
        ("incomplete", b"aaaaa\xc3", "UTF-8"),
    ],
)
async def test_read_many_reports_element_errors_and_continues(
    tmp_path: Path, name: str, payload: bytes, message: str
) -> None:
    (tmp_path / name).write_bytes(payload)
    (tmp_path / "ok.txt").write_text("fine")

    result = await ReadManyTool().execute(
        context(tmp_path),
        {"paths": [name, "ok.txt"], "max_bytes_per_file": 5},
    )

    assert message in result["files"][0]["error"]
    assert result["files"][1] == {"path": "ok.txt", "content": "fine", "truncated": False}


async def test_read_many_reports_nul_path_and_continues(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("fine")

    result = await ReadManyTool().execute(
        context(tmp_path),
        {"paths": ["bad\x00name", "ok.txt"], "max_bytes_per_file": 100},
    )

    assert "error" in result["files"][0]
    assert result["files"][1] == {"path": "ok.txt", "content": "fine", "truncated": False}


async def test_read_many_rejects_fifo_and_continues(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("fine")
    os.mkfifo(tmp_path / "pipe")

    result = await ReadManyTool().execute(
        context(tmp_path),
        {"paths": ["pipe", "ok.txt"], "max_bytes_per_file": 100},
    )

    assert "regular" in result["files"][0]["error"]
    assert result["files"][1] == {"path": "ok.txt", "content": "fine", "truncated": False}


async def test_read_many_rejects_symlink_without_leak(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "secret.txt").write_text("TOP SECRET content")
    (project / "link.txt").symlink_to(project / "secret.txt")
    (project / "ok.txt").write_text("fine")

    result = await ReadManyTool().execute(
        context(project),
        {"paths": ["link.txt", "ok.txt"], "max_bytes_per_file": 100},
    )

    assert "error" in result["files"][0]
    assert "TOP SECRET content" not in result["files"][0]["error"]
    assert "TOP SECRET content" not in str(result)
    assert result["files"][1] == {"path": "ok.txt", "content": "fine", "truncated": False}


async def test_read_many_rejects_symlink_swapped_after_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink planted after the path check must not leak external bytes."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "doc.txt").write_text("safe content")
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP SECRET content")
    (project / "ok.txt").write_text("fine")
    real_resolve = PathGuard.resolve

    def resolve_then_swap(
        self: PathGuard,
        value: str,
        *,
        must_exist: bool = True,
        allow_sensitive: bool = False,
    ) -> Path:
        resolved = real_resolve(self, value, must_exist=must_exist, allow_sensitive=allow_sensitive)
        if "doc" in value:
            (project / "doc.txt").unlink()
            (project / "doc.txt").symlink_to(outside)
        return resolved

    monkeypatch.setattr(PathGuard, "resolve", resolve_then_swap)

    result = await ReadManyTool().execute(
        context(project),
        {"paths": ["doc.txt", "ok.txt"], "max_bytes_per_file": 100},
    )

    assert "error" in result["files"][0]
    assert "TOP SECRET content" not in str(result)
    assert result["files"][1] == {"path": "ok.txt", "content": "fine", "truncated": False}


async def test_read_many_reports_cyclic_symlink_and_continues(tmp_path: Path) -> None:
    """A symlink loop on one element must not abort the whole batch."""
    project = tmp_path / "project"
    project.mkdir()
    loop = project / "loop"
    loop.symlink_to("loop")
    (project / "ok.txt").write_text("fine")

    result = await ReadManyTool().execute(
        context(project),
        {"paths": ["loop", "ok.txt"], "max_bytes_per_file": 100},
    )

    assert "error" in result["files"][0]
    assert result["files"][1] == {"path": "ok.txt", "content": "fine", "truncated": False}


async def test_multi_edit_rejects_exponential_growth(tmp_path: Path) -> None:
    """A doubling chain must be rejected before a huge allocation, leaving the file intact."""
    (tmp_path / "note.txt").write_text("x")
    edits = [{"old_text": "x", "new_text": "xx", "replace_all": True}] * 20

    with pytest.raises(ToolError) as excinfo:
        await MultiEditTool().execute(
            context(tmp_path),
            {"path": "note.txt", "edits": edits},
        )

    assert "Edit 20" in str(excinfo.value)
    assert (tmp_path / "note.txt").read_text() == "x"


async def test_multi_edit_applies_dependent_chain(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha")

    result = await MultiEditTool().execute(
        context(tmp_path),
        {
            "path": "note.txt",
            "edits": [
                {"old_text": "alpha", "new_text": "beta"},
                {"old_text": "beta", "new_text": "gamma"},
            ],
        },
    )

    assert (tmp_path / "note.txt").read_text() == "gamma"
    assert result == {"path": "note.txt", "edits_applied": 2, "replacements": 2}


async def test_multi_edit_failure_is_atomic_and_reports_edit_number(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha beta")

    with pytest.raises(ToolError) as excinfo:
        await MultiEditTool().execute(
            context(tmp_path),
            {
                "path": "note.txt",
                "edits": [
                    {"old_text": "alpha", "new_text": "ALPHA"},
                    {"old_text": "missing", "new_text": "x"},
                ],
            },
        )

    assert "2" in str(excinfo.value)
    assert (tmp_path / "note.txt").read_text() == "alpha beta"


async def test_multi_edit_rejects_ambiguous_old_text(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("a a a")

    with pytest.raises(ToolError) as excinfo:
        await MultiEditTool().execute(
            context(tmp_path),
            {"path": "note.txt", "edits": [{"old_text": "a", "new_text": "b"}]},
        )

    assert "Edit 1" in str(excinfo.value)
    assert (tmp_path / "note.txt").read_text() == "a a a"


async def test_multi_edit_detects_conflict_with_external_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "note.txt").write_text("alpha beta")
    real_file_stat = SafeProjectFS._file_stat
    state = {"calls": 0}

    def external_edit_before_replace(
        self: SafeProjectFS, name: str, parent_fd: int
    ) -> os.stat_result | None:
        state["calls"] += 1
        if state["calls"] == 2:
            (tmp_path / "note.txt").write_text("alpha beta external")
        return real_file_stat(self, name, parent_fd)

    monkeypatch.setattr(SafeProjectFS, "_file_stat", external_edit_before_replace)

    with pytest.raises(ToolError, match="conflict"):
        await MultiEditTool().execute(
            context(tmp_path),
            {"path": "note.txt", "edits": [{"old_text": "alpha", "new_text": "ALPHA"}]},
        )

    assert (tmp_path / "note.txt").read_text() == "alpha beta external"


def test_registry_gates_multi_edit_to_dev_mode() -> None:
    registry = ToolRegistry()

    assert registry.requires_confirmation("multi_edit", "dev") is True
    assert registry.requires_confirmation("glob", "dev") is False
    assert registry.requires_confirmation("read_many", "dev") is False

    registry.get("multi_edit", "dev")
    registry.get("glob", "dev")
    registry.get("read_many", "dev")
    with pytest.raises(ToolError):
        registry.get("multi_edit", "ask")

    registry.get("glob", "ask")
    registry.get("read_many", "ask")
    assert registry.requires_confirmation("glob", "ask") is False
    assert registry.requires_confirmation("read_many", "ask") is False


async def test_multi_edit_rejects_lone_surrogate(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    arguments = {"path": "a.txt", "edits": [{"old_text": "x", "new_text": "\ud800"}]}
    with pytest.raises(ToolError, match="Edit 1"):
        await MultiEditTool().execute(context(tmp_path), arguments)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"

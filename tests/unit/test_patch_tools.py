from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.safe_fs import SafeProjectFS
from agent.tools.base import ToolContext, ToolError
from agent.tools.patch_tools import (
    ApplyPatchTool,
    EditByLinesTool,
    Hunk,
    apply_hunks,
    parse_unified_diff,
)


def _change_diff() -> str:
    return """diff --git a/hello.txt b/hello.txt
index 1111111..2222222 100644
--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
 hello
-world
+universe
"""


def _create_diff() -> str:
    return """diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+line1
+line2
"""


def _delete_diff() -> str:
    return """diff --git a/old.txt b/old.txt
deleted file mode 100644
index 4444444..0000000
--- a/old.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-gone1
-gone2
"""


def test_parse_change_basic() -> None:
    patches = parse_unified_diff(_change_diff())
    assert len(patches) == 1
    patch = patches[0]
    assert patch.old_path == "hello.txt"
    assert patch.new_path == "hello.txt"
    assert len(patch.hunks) == 1
    hunk = patch.hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_lines == ["hello\n", "world\n"]
    assert hunk.new_lines == ["hello\n", "universe\n"]


def test_parse_creation_uses_dev_null() -> None:
    patches = parse_unified_diff(_create_diff())
    patch = patches[0]
    assert patch.old_path is None
    assert patch.new_path == "new.txt"
    hunk = patch.hunks[0]
    assert hunk.old_start == 0
    assert hunk.old_lines == []
    assert hunk.new_lines == ["line1\n", "line2\n"]


def test_parse_deletion_uses_dev_null() -> None:
    patches = parse_unified_diff(_delete_diff())
    patch = patches[0]
    assert patch.old_path == "old.txt"
    assert patch.new_path is None


def test_parse_multiple_files() -> None:
    text = _change_diff() + "\n" + _create_diff()
    patches = parse_unified_diff(text)
    assert [p.old_path for p in patches] == ["hello.txt", None]
    assert [p.new_path for p in patches] == ["hello.txt", "new.txt"]


def test_parse_strips_a_b_prefixes_without_timestamps() -> None:
    text = """--- a/dir/file.txt
+++ b/dir/file.txt
@@ -1 +1 @@
-a
+b
"""
    patch = parse_unified_diff(text)[0]
    assert patch.old_path == "dir/file.txt"
    assert patch.new_path == "dir/file.txt"


def test_parse_strips_timestamp_after_tab() -> None:
    text = "--- a/f.txt\t2024-01-01 00:00:00\n+++ b/f.txt\t2024-01-01 00:00:01\n@@ -1 +1 @@\n x\n"
    patch = parse_unified_diff(text)[0]
    assert patch.old_path == "f.txt"
    assert patch.new_path == "f.txt"


def test_parse_empty_text_raises() -> None:
    with pytest.raises(ToolError):
        parse_unified_diff("")


def test_parse_whitespace_only_raises() -> None:
    with pytest.raises(ToolError):
        parse_unified_diff("   \n\t\n  \n")


def test_parse_no_files_raises() -> None:
    text = "diff --git a/x b/x\nindex 1..2 100644\n"
    with pytest.raises(ToolError):
        parse_unified_diff(text)


def test_parse_bad_count_raises() -> None:
    text = """--- a/f.txt
+++ b/f.txt
@@ -1,5 +1,3 @@
 a
 b
"""
    with pytest.raises(ToolError) as exc:
        parse_unified_diff(text)
    assert "Malformed hunk" in str(exc.value)


def test_parse_missing_plus_plus_raises() -> None:
    with pytest.raises(ToolError):
        parse_unified_diff("--- a/f.txt\n@@ -1 +1 @@\nx\n")


def test_apply_single_hunk() -> None:
    original = "hello\nworld\n"
    hunk = Hunk(old_start=1, old_lines=["hello\n", "world\n"], new_lines=["hello\n", "universe\n"])
    assert apply_hunks(original, [hunk], path="hello.txt") == "hello\nuniverse\n"


def test_apply_two_hunks_with_shift() -> None:
    original = "one\ntwo\nthree\nfour\n"
    first = Hunk(old_start=1, old_lines=["one\n"], new_lines=["ONE\n", "extra\n"])
    second = Hunk(
        old_start=4,
        old_lines=["four\n"],
        new_lines=["FOUR\n"],
    )
    assert apply_hunks(original, [first, second], path="f.txt") == "ONE\nextra\ntwo\nthree\nFOUR\n"


def test_apply_context_mismatch_raises_hunk_number() -> None:
    original = "one\ntwo\n"
    hunk = Hunk(old_start=1, old_lines=["nope\n"], new_lines=["x\n"])
    with pytest.raises(ToolError) as exc:
        apply_hunks(original, [hunk], path="f.txt")
    assert "hunk 1 does not apply" in str(exc.value)
    assert "f.txt" in str(exc.value)


def test_apply_overlapping_hunks_raises() -> None:
    original = "a\nb\nc\n"
    first = Hunk(old_start=1, old_lines=["a\n", "b\n"], new_lines=["A\n"])
    second = Hunk(old_start=2, old_lines=["b\n", "c\n"], new_lines=["C\n"])
    with pytest.raises(ToolError):
        apply_hunks(original, [first, second], path="f.txt")


def test_apply_old_file_without_trailing_newline() -> None:
    original = "a\nb"
    hunk = Hunk(old_start=2, old_lines=["b"], new_lines=["b", "c\n"])
    assert apply_hunks(original, [hunk], path="f.txt") == "a\nbc\n"


def test_apply_new_file_without_trailing_newline() -> None:
    original = "a\nb\n"
    hunk = Hunk(old_start=2, old_lines=["b\n"], new_lines=["b\n", "c"])
    assert apply_hunks(original, [hunk], path="f.txt") == "a\nb\nc"


def test_apply_create_file_from_empty_text() -> None:
    hunk = Hunk(old_start=0, old_lines=[], new_lines=["line1\n", "line2\n"])
    assert apply_hunks("", [hunk], path="new.txt") == "line1\nline2\n"


def test_apply_pure_insertion_lands_after_line_n() -> None:
    # `diff -U0` pure insertion: old_lines is empty and old_start = N > 0 means
    # the lines are inserted AFTER line N, i.e. at 0-based index N (not N-1).
    original = "a\nb\nc\n"
    hunk = Hunk(old_start=2, old_lines=[], new_lines=["X\n", "Y\n"])
    assert apply_hunks(original, [hunk], path="f.txt") == "a\nb\nX\nY\nc\n"


def test_apply_pure_insertion_at_head_and_tail() -> None:
    original = "a\nb\n"
    # Insertion at the very start uses old_start == 0; old_start == N inserts
    # after line N.
    head = Hunk(old_start=0, old_lines=[], new_lines=["H\n"])
    assert apply_hunks(original, [head], path="f.txt") == "H\na\nb\n"
    tail = Hunk(old_start=2, old_lines=[], new_lines=["T\n"])
    assert apply_hunks(original, [tail], path="f.txt") == "a\nb\nT\n"


def test_apply_pure_insertion_then_modification_shift() -> None:
    original = "a\nb\nc\n"
    insert = Hunk(old_start=2, old_lines=[], new_lines=["X\n", "Y\n"])
    modify = Hunk(old_start=3, old_lines=["c\n"], new_lines=["C\n"])
    # Insertion adds 2 lines after b, shifting the later hunk's position by +2.
    assert apply_hunks(original, [insert, modify], path="f.txt") == "a\nb\nX\nY\nC\n"


def _context(root: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=root)


def _modify_diff() -> str:
    return "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,2 +1,2 @@\n hello\n-world\n+universe\n"


def _create_diff() -> str:
    return "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,2 @@\n+line1\n+line2\n"


def _delete_diff() -> str:
    return "--- a/old.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-gone1\n-gone2\n"


async def test_apply_patch_create(tmp_path: Path) -> None:
    result = await ApplyPatchTool().execute(_context(tmp_path), {"patch": _create_diff()})
    assert result == {"files": [{"path": "new.txt", "action": "create", "hunks": 1}]}
    assert (tmp_path / "new.txt").read_text() == "line1\nline2\n"


async def test_apply_patch_modify(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello\nworld\n")
    result = await ApplyPatchTool().execute(_context(tmp_path), {"patch": _modify_diff()})
    assert result == {"files": [{"path": "hello.txt", "action": "modify", "hunks": 1}]}
    assert (tmp_path / "hello.txt").read_text() == "hello\nuniverse\n"


async def test_apply_patch_delete(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("gone1\ngone2\n")
    result = await ApplyPatchTool().execute(_context(tmp_path), {"patch": _delete_diff()})
    assert result == {"files": [{"path": "old.txt", "action": "delete", "hunks": 1}]}
    assert not (tmp_path / "old.txt").exists()


async def test_apply_patch_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello\nworld\n")
    (tmp_path / "old.txt").write_text("gone1\ngone2\n")
    patch = _modify_diff() + _create_diff() + _delete_diff()
    result = await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    assert result["files"] == [
        {"path": "hello.txt", "action": "modify", "hunks": 1},
        {"path": "new.txt", "action": "create", "hunks": 1},
        {"path": "old.txt", "action": "delete", "hunks": 1},
    ]
    assert (tmp_path / "hello.txt").read_text() == "hello\nuniverse\n"
    assert (tmp_path / "new.txt").read_text() == "line1\nline2\n"
    assert not (tmp_path / "old.txt").exists()


async def test_apply_patch_mismatch_in_second_file_changes_nothing(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("one\n")
    (tmp_path / "second.txt").write_text("different\n")
    patch = (
        "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-nope\n+YES\n"
    )
    with pytest.raises(ToolError) as exc:
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    assert "second.txt" in str(exc.value)
    # Nothing changed: the first file's successful hunk was never written.
    assert (tmp_path / "first.txt").read_text() == "one\n"


async def test_apply_patch_rejects_parent_traversal(tmp_path: Path) -> None:
    patch = "--- a/../x\n+++ b/../x\n@@ -1 +1 @@\n-a\n+b\n"
    with pytest.raises(ToolError) as exc:
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    assert "Parent traversal" in str(exc.value)


async def test_apply_patch_rejects_absolute_path(tmp_path: Path) -> None:
    patch = "--- /tmp/abs\n+++ /tmp/abs\n@@ -1 +1 @@\n-a\n+b\n"
    with pytest.raises(ToolError):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})


async def test_apply_patch_create_existing_file_fails(tmp_path: Path) -> None:
    (tmp_path / "new.txt").write_text("already here\n")
    with pytest.raises(ToolError) as exc:
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": _create_diff()})
    assert "already exists" in str(exc.value)
    assert (tmp_path / "new.txt").read_text() == "already here\n"


async def test_apply_patch_create_then_repeat_is_idempotent_failure(tmp_path: Path) -> None:
    # Running the same create patch twice must not create a duplicate file.
    for _ in range(2):
        if (tmp_path / "new.txt").exists():
            with pytest.raises(ToolError) as exc:
                await ApplyPatchTool().execute(_context(tmp_path), {"patch": _create_diff()})
            assert "already exists" in str(exc.value)
        else:
            await ApplyPatchTool().execute(_context(tmp_path), {"patch": _create_diff()})
    assert (tmp_path / "new.txt").read_text() == "line1\nline2\n"


async def test_apply_patch_deletion_does_not_empty_file_fails(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("gone1\ngone2\nextra\n")
    # Hunk only removes the first two lines, so the result is not empty.
    with pytest.raises(ToolError) as exc:
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": _delete_diff()})
    assert "did not empty" in str(exc.value)
    assert (tmp_path / "old.txt").read_text() == "gone1\ngone2\nextra\n"


async def test_apply_patch_empty_and_whitespace_patch_fails(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": ""})
    with pytest.raises(ToolError):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": "   \n\t\n"})


async def test_apply_patch_overlong_patch_fails(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": "x" * 200_001})


async def test_apply_patch_missing_file_modify_fails_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as exc:
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": _modify_diff()})
    assert "does not exist" in str(exc.value)


async def test_apply_patch_symlink_file_fails(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "victim.txt"
    (project / "note.txt").symlink_to(outside)
    patch = "--- a/note.txt\n+++ b/note.txt\n@@ -1 +1 @@\n-a\n+b\n"
    with pytest.raises(ToolError, match="Symlink"):
        await ApplyPatchTool().execute(_context(project), {"patch": patch})
    assert not outside.exists() or outside.read_text() == ""


async def test_edit_by_lines_replace(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\nthree\n")
    result = await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "f.txt", "start_line": 2, "end_line": 2, "new_text": "TWO\n"},
    )
    assert result == {"path": "f.txt", "replaced_lines": 1, "inserted_lines": 1}
    assert (tmp_path / "f.txt").read_text() == "one\nTWO\nthree\n"


async def test_edit_by_lines_insert_before_line(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    result = await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "f.txt", "start_line": 2, "end_line": 1, "new_text": "mid\n"},
    )
    assert result == {"path": "f.txt", "replaced_lines": 0, "inserted_lines": 1}
    assert (tmp_path / "f.txt").read_text() == "one\nmid\ntwo\n"


async def test_edit_by_lines_insert_at_head_and_tail(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\n")
    await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "f.txt", "start_line": 1, "end_line": 0, "new_text": "head\n"},
    )
    assert (tmp_path / "f.txt").read_text() == "head\none\n"
    await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "f.txt", "start_line": 3, "end_line": 2, "new_text": "tail\n"},
    )
    assert (tmp_path / "f.txt").read_text() == "head\none\ntail\n"


async def test_edit_by_lines_delete_range(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\nthree\nfour\n")
    result = await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "f.txt", "start_line": 2, "end_line": 3, "new_text": ""},
    )
    assert result == {"path": "f.txt", "replaced_lines": 2, "inserted_lines": 0}
    assert (tmp_path / "f.txt").read_text() == "one\nfour\n"


async def test_edit_by_lines_expected_old_text_matches(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    result = await EditByLinesTool().execute(
        _context(tmp_path),
        {
            "path": "f.txt",
            "start_line": 1,
            "end_line": 2,
            "new_text": "ONE\nTWO\n",
            "expected_old_text": "one\ntwo\n",
        },
    )
    assert result["path"] == "f.txt"
    assert (tmp_path / "f.txt").read_text() == "ONE\nTWO\n"


async def test_edit_by_lines_expected_old_text_mismatch_fails(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    with pytest.raises(ToolError) as exc:
        await EditByLinesTool().execute(
            _context(tmp_path),
            {
                "path": "f.txt",
                "start_line": 1,
                "end_line": 2,
                "new_text": "X\n",
                "expected_old_text": "different\n",
            },
        )
    assert "expected_old_text" in str(exc.value)
    assert (tmp_path / "f.txt").read_text() == "one\ntwo\n"


async def test_edit_by_lines_boundaries(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a\nb\n")
    # Replacement ranging past the last line must fail.
    with pytest.raises(ToolError):
        await EditByLinesTool().execute(
            _context(tmp_path), {"path": "f.txt", "start_line": 2, "end_line": 3, "new_text": "x\n"}
        )
    # Inconsistent lines (end_line < start_line - 1) must fail.
    with pytest.raises(ToolError):
        await EditByLinesTool().execute(
            _context(tmp_path), {"path": "f.txt", "start_line": 3, "end_line": 1, "new_text": "x\n"}
        )
    # start_line of 0 must fail.
    with pytest.raises(ToolError):
        await EditByLinesTool().execute(
            _context(tmp_path), {"path": "f.txt", "start_line": 0, "end_line": 0, "new_text": "x\n"}
        )


async def test_edit_by_lines_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        await EditByLinesTool().execute(
            _context(tmp_path),
            {"path": "missing.txt", "start_line": 1, "end_line": 1, "new_text": "x\n"},
        )


def test_omitted_hunk_count_means_one_line() -> None:
    with pytest.raises(ToolError, match="Malformed hunk"):
        parse_unified_diff("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n-b\n+c\n")
    (patch,) = parse_unified_diff("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+c\n")
    assert patch.hunks[0].old_lines == ["a\n"] and patch.hunks[0].new_lines == ["c\n"]


async def test_edit_by_lines_on_empty_file(tmp_path: Path) -> None:
    """Inserting lines into an existing empty file must work (defect 1)."""
    (tmp_path / "empty.txt").touch()
    result = await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "empty.txt", "start_line": 1, "end_line": 0, "new_text": "hello\n"},
    )
    assert result == {"path": "empty.txt", "replaced_lines": 0, "inserted_lines": 1}
    assert (tmp_path / "empty.txt").read_text() == "hello\n"


async def test_apply_patch_create_in_missing_subdir(tmp_path: Path) -> None:
    """Creating a file in a non-existent subdirectory must create parents (defect 2)."""
    patch = "--- /dev/null\n+++ b/dir/file.txt\n@@ -0,0 +1,1 @@\n+content\n"
    result = await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    assert result == {"files": [{"path": "dir/file.txt", "action": "create", "hunks": 1}]}
    assert (tmp_path / "dir/file.txt").read_text() == "content\n"


async def test_edit_by_lines_insert_after_last_line_no_trailing_newline(
    tmp_path: Path,
) -> None:
    """Insertion after the final line of a file without trailing newline (defect 3)."""
    (tmp_path / "f.txt").write_text("a\nb")
    await EditByLinesTool().execute(
        _context(tmp_path),
        {"path": "f.txt", "start_line": 3, "end_line": 2, "new_text": "c\n"},
    )
    assert (tmp_path / "f.txt").read_text() == "a\nb\nc\n"


def test_apply_insertion_beyond_file_end_raises() -> None:
    """A pure insertion past the end of the file must fail (defect 4)."""
    original = "a\nb\n"
    hunk = Hunk(old_start=5, old_lines=[], new_lines=["X\n"])
    with pytest.raises(ToolError):
        apply_hunks(original, [hunk], path="f.txt")


async def test_edit_by_lines_toctou_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent change between the first read and the write must fail (defect 5)."""
    target = tmp_path / "f.txt"
    target.write_text("one\ntwo\n")
    real_read_regular = SafeProjectFS._read_regular
    calls = {"count": 0}

    def change_between_reads(
        self: SafeProjectFS, name: str, parent_fd: int, max_bytes: int
    ) -> tuple[str, os.stat_result]:
        calls["count"] += 1
        if calls["count"] == 2:
            # This is the read inside replace_if_unchanged; modify first.
            target.write_text("one\nTWO\nCHANGED\n")
        return real_read_regular(self, name, parent_fd, max_bytes)

    monkeypatch.setattr(SafeProjectFS, "_read_regular", change_between_reads)

    with pytest.raises(ToolError, match="conflict"):
        await EditByLinesTool().execute(
            _context(tmp_path),
            {"path": "f.txt", "start_line": 2, "end_line": 2, "new_text": "OOO\n"},
        )
    # The externally-changed content must be left untouched.
    assert target.read_text() == "one\nTWO\nCHANGED\n"


async def test_apply_patch_modify_between_passes_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file changed between pass one and pass two must not be overwritten."""
    target = tmp_path / "hello.txt"
    target.write_text("hello\nworld\n")
    real_replace = SafeProjectFS.replace_if_unchanged
    changed = {"flag": False}

    def change_before_replace(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        if not changed["flag"] and path == "hello.txt":
            changed["flag"] = True
            # Simulate an external edit after pass one read the file.
            target.write_text("hello\nWORLD-CHANGED\n")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", change_before_replace)

    with pytest.raises(ToolError, match="conflict"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": _modify_diff()})
    # The external content must be preserved, not overwritten by the patch.
    assert target.read_text() == "hello\nWORLD-CHANGED\n"


async def test_apply_patch_create_between_passes_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file created between pass one and pass two must not be overwritten."""
    real_create = SafeProjectFS.create_text_exclusive

    def create_after_check(self: SafeProjectFS, path: str, content: str) -> Path:
        # Simulate a concurrent file appearing after pass one checked it.
        (tmp_path / "new.txt").write_text("someone else\n")
        return real_create(self, path, content)

    monkeypatch.setattr(SafeProjectFS, "create_text_exclusive", create_after_check)

    with pytest.raises(ToolError, match="already exists"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": _create_diff()})
    # The foreign file must remain untouched.
    assert (tmp_path / "new.txt").read_text() == "someone else\n"


async def test_apply_patch_second_file_failure_restores_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write failure on the second file must roll back the first file."""
    (tmp_path / "first.txt").write_text("one\n")
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged
    calls = {"count": 0}

    def fail_on_second(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        calls["count"] += 1
        if calls["count"] == 2:
            raise ToolError("injected write failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", fail_on_second)

    with pytest.raises(ToolError, match="injected write failure"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The first file must be rolled back to its original content.
    assert (tmp_path / "first.txt").read_text() == "one\n"
    assert (tmp_path / "second.txt").read_text() == "two\n"


async def test_apply_patch_deleted_file_restored_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file deleted in pass two must be restored when a later file fails."""
    (tmp_path / "old.txt").write_text("gone1\ngone2\n")
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- a/old.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-gone1\n-gone2\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged

    def fail_on_second(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        if path == "second.txt":
            raise ToolError("injected write failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", fail_on_second)

    with pytest.raises(ToolError, match="injected write failure"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The deleted file must be recreated with its original content.
    assert (tmp_path / "old.txt").read_text() == "gone1\ngone2\n"
    assert (tmp_path / "second.txt").read_text() == "two\n"


async def test_apply_patch_created_file_removed_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file created in pass two must be removed when a later file fails."""
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,1 @@\n+content\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged

    def fail_on_second(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        if path == "second.txt":
            raise ToolError("injected write failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", fail_on_second)

    with pytest.raises(ToolError, match="injected write failure"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The created file must be removed during rollback.
    assert not (tmp_path / "new.txt").exists()
    assert (tmp_path / "second.txt").read_text() == "two\n"


async def test_apply_patch_rollback_failure_lists_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback failure must report which files remain changed."""
    (tmp_path / "first.txt").write_text("one\n")
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged
    calls = {"count": 0}

    def fail_and_break_rollback(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        calls["count"] += 1
        if calls["count"] == 2:
            raise ToolError("injected write failure")
        if calls["count"] == 3:
            # Rollback of the first file itself fails.
            raise ToolError("injected rollback failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", fail_and_break_rollback)

    with pytest.raises(ToolError, match="Rollback failed for files"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The first file remains changed because the rollback failed.
    assert (tmp_path / "first.txt").read_text() == "ONE\n"


async def test_apply_patch_cyrillic_size_limit_counts_bytes(tmp_path: Path) -> None:
    """The patch size limit must count UTF-8 bytes, not characters."""
    cyrillic_excess = "я" * 100_001  # 100_001 chars = 200_002 bytes (2 bytes per char).
    with pytest.raises(ToolError, match="bytes"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": cyrillic_excess})

    # Exactly 100_000 Cyrillic chars = 200_000 bytes, within the limit.
    cyrillic_boundary = "я" * 100_000
    with pytest.raises(ToolError, match="Malformed patch"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": cyrillic_boundary})


async def test_apply_patch_symlink_between_passes_triggers_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink planted during pass two must roll back earlier files (defect D1)."""
    outside = tmp_path / "victim.txt"
    outside.write_text("secret\n")
    (tmp_path / "first.txt").write_text("one\n")
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged
    calls = {"count": 0}

    def plant_symlink(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        calls["count"] += 1
        if calls["count"] == 2:
            # Replace the second file with a symlink between passes.
            (tmp_path / "second.txt").unlink()
            (tmp_path / "second.txt").symlink_to(outside)
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", plant_symlink)

    with pytest.raises(ToolError):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The first file must be rolled back even though the second error is a
    # PathSecurityError, not a ToolError.
    assert (tmp_path / "first.txt").read_text() == "one\n"
    assert (tmp_path / "second.txt").is_symlink()
    assert outside.read_text() == "secret\n"


async def test_apply_patch_rollback_does_not_overwrite_recreated_deleted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback must not overwrite a deleted file recreated by another party (defect D2)."""
    (tmp_path / "old.txt").write_text("gone1\ngone2\n")
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- a/old.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-gone1\n-gone2\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged

    def recreate_before_fail(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        if path == "second.txt":
            # The deleted file is recreated with foreign content before rollback.
            (tmp_path / "old.txt").write_text("someone else\n")
            raise ToolError("injected write failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", recreate_before_fail)

    with pytest.raises(ToolError, match="Rollback failed for files"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The foreign content must be preserved, not overwritten by rollback.
    assert (tmp_path / "old.txt").read_text() == "someone else\n"


async def test_apply_patch_rollback_does_not_delete_replaced_created_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback must not delete a created file replaced by another party (defect D3)."""
    (tmp_path / "second.txt").write_text("two\n")
    patch = (
        "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,1 @@\n+content\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = SafeProjectFS.replace_if_unchanged

    def replace_before_fail(self: SafeProjectFS, path: str, expected: str, new: str) -> Path:
        if path == "second.txt":
            # The created file is replaced with foreign content before rollback.
            (tmp_path / "new.txt").write_text("someone else\n")
            raise ToolError("injected write failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(SafeProjectFS, "replace_if_unchanged", replace_before_fail)

    with pytest.raises(ToolError, match="Rollback failed for files"):
        await ApplyPatchTool().execute(_context(tmp_path), {"patch": patch})
    # The foreign file must be preserved, not deleted by rollback.
    assert (tmp_path / "new.txt").read_text() == "someone else\n"

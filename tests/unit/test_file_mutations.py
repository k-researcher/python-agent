from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from agent.safe_fs import SafeProjectFS
from agent.security import PathSecurityError
from agent.tools.base import ToolError


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def filesystem(project: Path) -> SafeProjectFS:
    return SafeProjectFS(project)


def run_operation(filesystem: SafeProjectFS, operation: str, path: str) -> object:
    if operation == "write":
        return filesystem.write_text(path, "new")
    if operation == "edit":
        return filesystem.edit_text(path, "old", "new", replace_all=False)
    if operation == "read":
        return filesystem.read_text(path, max_bytes=1_000_000)
    return filesystem.remove_file(path)


def test_write_refuses_dangling_symlink_escape(
    project: Path, filesystem: SafeProjectFS, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    link = project / "notes.txt"
    link.symlink_to(outside)

    with pytest.raises(PathSecurityError, match="Symlink"):
        filesystem.write_text("notes.txt", "new")

    assert link.is_symlink()
    assert not outside.exists()
    assert not list(project.glob("*.tmp"))


@pytest.mark.parametrize("operation", ["write", "edit", "read", "remove"])
@pytest.mark.parametrize("internal", [True, False])
def test_refuses_symlink_directory_components(
    project: Path, filesystem: SafeProjectFS, tmp_path: Path, operation: str, internal: bool
) -> None:
    directory = (project if internal else tmp_path) / "target"
    directory.mkdir()
    target = directory / "notes.txt"
    target.write_text("old", encoding="utf-8")
    (project / "link").symlink_to(directory, target_is_directory=True)

    with pytest.raises(PathSecurityError, match="Symlink"):
        run_operation(filesystem, operation, "link/notes.txt")

    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(item.name for item in directory.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize("operation", ["write", "edit", "read", "remove"])
def test_refuses_final_internal_symlink(
    project: Path, filesystem: SafeProjectFS, operation: str
) -> None:
    target = project / "target.txt"
    target.write_text("old", encoding="utf-8")
    link = project / "link.txt"
    link.symlink_to(target)

    with pytest.raises(PathSecurityError, match="Symlink"):
        run_operation(filesystem, operation, "link.txt")

    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "old"


def test_creates_nested_directories(project: Path, filesystem: SafeProjectFS) -> None:
    relative = filesystem.write_text("src/new/module.py", "print('ok')\n")

    assert relative == Path("src/new/module.py")
    assert (project / relative).read_text(encoding="utf-8") == "print('ok')\n"
    assert stat.S_IMODE((project / relative).stat().st_mode) == 0o600
    assert sorted(item.name for item in (project / "src/new").iterdir()) == ["module.py"]


def test_can_disable_parent_creation(project: Path, filesystem: SafeProjectFS) -> None:
    with pytest.raises(ToolError):
        filesystem.write_text("missing/notes.txt", "new", create_parents=False)

    assert not (project / "missing").exists()


@pytest.mark.parametrize("operation", ["write", "edit"])
def test_preserves_executable_permissions(
    project: Path, filesystem: SafeProjectFS, operation: str
) -> None:
    target = project / "run.sh"
    target.write_text("echo old\n", encoding="utf-8")
    target.chmod(0o755)

    run_operation(filesystem, operation, "run.sh")

    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert "new" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("operation", ["write", "edit"])
def test_file_fsync_failure_preserves_original_and_removes_temp(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")
    original = target.stat()

    def fail_fsync(descriptor: int) -> None:
        assert stat.S_ISREG(os.fstat(descriptor).st_mode)
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(ToolError, match="injected fsync failure"):
        run_operation(filesystem, operation, "notes.txt")

    assert target.read_text(encoding="utf-8") == "old"
    assert target.stat().st_ino == original.st_ino
    assert sorted(item.name for item in project.iterdir()) == ["notes.txt"]


def test_new_file_fsync_failure_leaves_no_file(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(ToolError, match="injected fsync failure"):
        filesystem.write_text("notes.txt", "new")

    assert not list(project.iterdir())


def test_directory_fsync_is_best_effort(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = os.fsync
    synced: list[str] = []

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            synced.append("directory")
            raise OSError("directory fsync is unsupported")
        synced.append("file")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    assert filesystem.write_text("notes.txt", "new") == Path("notes.txt")
    assert (project / "notes.txt").read_text(encoding="utf-8") == "new"
    assert synced == ["file", "directory"]


@pytest.mark.parametrize("change", ["inode", "size", "mtime", "remove"])
def test_edit_detects_conflict_before_replace(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")
    original = target.stat()
    real_fsync = os.fsync

    def change_before_replace(descriptor: int) -> None:
        real_fsync(descriptor)
        if change == "inode":
            replacement = project / "replacement.txt"
            replacement.write_text("old", encoding="utf-8")
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            replacement.replace(target)
        elif change == "size":
            target.write_text("concurrent update", encoding="utf-8")
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        elif change == "mtime":
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
        else:
            target.unlink()

    monkeypatch.setattr(os, "fsync", change_before_replace)

    with pytest.raises(ToolError, match="conflict"):
        filesystem.edit_text("notes.txt", "old", "new", replace_all=False)

    if change == "remove":
        assert not target.exists()
        assert not list(project.iterdir())
    else:
        expected = "concurrent update" if change == "size" else "old"
        assert target.read_text(encoding="utf-8") == expected
        assert sorted(item.name for item in project.iterdir()) == ["notes.txt"]


def test_write_rechecks_final_symlink_before_replace(
    project: Path, filesystem: SafeProjectFS, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    real_fsync = os.fsync

    def plant_symlink(descriptor: int) -> None:
        real_fsync(descriptor)
        target.unlink()
        target.symlink_to(outside)

    monkeypatch.setattr(os, "fsync", plant_symlink)

    with pytest.raises(PathSecurityError, match="Symlink"):
        filesystem.write_text("notes.txt", "new")

    assert target.is_symlink()
    assert not outside.exists()
    assert sorted(item.name for item in project.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize(
    ("old", "replace_all", "message"),
    [("", False, "not found"), ("missing", True, "not found"), ("old", False, "occurs 2 times")],
)
def test_edit_preserves_fragment_errors(
    project: Path, filesystem: SafeProjectFS, old: str, replace_all: bool, message: str
) -> None:
    target = project / "notes.txt"
    target.write_text("old old", encoding="utf-8")

    with pytest.raises(ToolError, match=message):
        filesystem.edit_text("notes.txt", old, "new", replace_all=replace_all)

    assert target.read_text(encoding="utf-8") == "old old"
    assert sorted(item.name for item in project.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize(
    ("content", "replace_all", "count"), [("old", False, 1), ("old old", True, 2)]
)
def test_edit_returns_relative_path_and_count(
    project: Path, filesystem: SafeProjectFS, content: str, replace_all: bool, count: int
) -> None:
    target = project / "notes.txt"
    target.write_text(content, encoding="utf-8")

    assert filesystem.edit_text(str(target), "old", "new", replace_all=replace_all) == (
        Path("notes.txt"),
        count,
    )
    assert target.read_text(encoding="utf-8") == content.replace("old", "new")


def test_remove_returns_relative_path(project: Path, filesystem: SafeProjectFS) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")

    assert filesystem.remove_file(str(target)) == Path("notes.txt")
    assert not target.exists()


@pytest.mark.parametrize("operation", ["write", "edit", "read", "remove"])
@pytest.mark.parametrize("path", [".env", "nested/.env.production", "keys/private.key"])
def test_blocks_sensitive_paths(filesystem: SafeProjectFS, operation: str, path: str) -> None:
    with pytest.raises(PathSecurityError, match="secret-bearing"):
        run_operation(filesystem, operation, path)


def test_allows_environment_template(filesystem: SafeProjectFS) -> None:
    assert filesystem.write_text(".env.example", "KEY=\n") == Path(".env.example")
    assert filesystem.read_text(".env.example", max_bytes=5) == "KEY=\n"


@pytest.mark.parametrize("path", ["../outside.txt", "a/../../outside.txt", "a/../notes.txt"])
def test_rejects_parent_traversal(filesystem: SafeProjectFS, path: str) -> None:
    with pytest.raises(PathSecurityError, match="traversal"):
        filesystem.write_text(path, "new")


def test_rejects_absolute_outside_path(filesystem: SafeProjectFS, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    with pytest.raises(PathSecurityError, match="outside"):
        filesystem.write_text(str(outside), "new")
    assert not outside.exists()


def test_normalizes_lexical_path(project: Path, filesystem: SafeProjectFS) -> None:
    assert filesystem.write_text(str(project / "notes.txt"), "new") == Path("notes.txt")
    assert filesystem.read_text("./notes.txt", max_bytes=3) == "new"


def test_rejects_nul(filesystem: SafeProjectFS) -> None:
    with pytest.raises(PathSecurityError, match="NUL"):
        filesystem.write_text("notes\0.txt", "new")


@pytest.mark.parametrize("operation", ["write", "edit", "read", "remove"])
def test_rejects_fifo(project: Path, filesystem: SafeProjectFS, operation: str) -> None:
    fifo = project / "pipe"
    os.mkfifo(fifo)

    with pytest.raises(ToolError, match="regular"):
        run_operation(filesystem, operation, "pipe")

    assert stat.S_ISFIFO(fifo.stat().st_mode)


@pytest.mark.parametrize("operation", ["write", "edit", "read", "remove"])
def test_rejects_directory(project: Path, filesystem: SafeProjectFS, operation: str) -> None:
    (project / "directory").mkdir()

    with pytest.raises(ToolError, match="regular"):
        run_operation(filesystem, operation, "directory")

    assert (project / "directory").is_dir()


def test_read_requires_utf8(project: Path, filesystem: SafeProjectFS) -> None:
    (project / "notes.txt").write_bytes(b"\xff")

    with pytest.raises(ToolError, match="UTF-8"):
        filesystem.read_text("notes.txt", max_bytes=10)


def test_read_limit_counts_bytes(project: Path, filesystem: SafeProjectFS) -> None:
    (project / "notes.txt").write_text("\u00e9", encoding="utf-8")

    assert filesystem.read_text("notes.txt", max_bytes=2) == "\u00e9"
    with pytest.raises(ToolError, match="larger than 1 bytes"):
        filesystem.read_text("notes.txt", max_bytes=1)


def test_edit_keeps_one_megabyte_limit(project: Path, filesystem: SafeProjectFS) -> None:
    (project / "notes.txt").write_text("x" * 1_000_001, encoding="utf-8")

    with pytest.raises(ToolError, match="larger than 1000000 bytes"):
        filesystem.edit_text("notes.txt", "x", "y", replace_all=True)


def test_refuses_missing_dir_fd_support(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd - {os.open})

    with pytest.raises(ToolError, match="POSIX"):
        SafeProjectFS(project)


def test_closes_descriptors_after_failure(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    closed: list[int] = []

    def track_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def track_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def fail_fsync(descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", track_close)
    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(ToolError, match="injected fsync failure"):
        filesystem.write_text("nested/deeper/notes.txt", "new")

    assert len(opened) == 4
    assert sorted(opened) == sorted(closed)
    assert not list(project.rglob("*.tmp"))


def test_handles_short_writes(filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch) -> None:
    real_write: Callable[[int, bytes | memoryview], int] = os.write

    def short_write(descriptor: int, data: memoryview) -> int:
        return real_write(descriptor, data[:1])

    monkeypatch.setattr(os, "write", short_write)

    filesystem.write_text("notes.txt", "new\u00e9")
    assert filesystem.read_text("notes.txt", max_bytes=5) == "new\u00e9"


def test_refuses_symlink_planted_after_parent_creation(
    project: Path, filesystem: SafeProjectFS, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdir = os.mkdir

    def replace_new_directory(path: str, *, dir_fd: int) -> None:
        real_mkdir(path, dir_fd=dir_fd)
        os.rmdir(path, dir_fd=dir_fd)
        os.symlink(str(outside), path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", replace_new_directory)

    with pytest.raises(PathSecurityError, match="Symlink"):
        filesystem.write_text("nested/notes.txt", "new")

    assert (project / "nested").is_symlink()
    assert not list(outside.iterdir())


def test_fifo_planted_between_stat_and_open_is_not_read(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")
    real_open = os.open

    def plant_fifo(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == "notes.txt":
            assert flags & os.O_NONBLOCK
            target.unlink()
            os.mkfifo(target)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", plant_fifo)

    with pytest.raises(ToolError, match="regular"):
        filesystem.read_text("notes.txt", max_bytes=10)


def test_read_rejects_growth_beyond_limit(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")
    real_read = os.read

    def grow_before_read(descriptor: int, size: int) -> bytes:
        target.write_text("old and more", encoding="utf-8")
        return real_read(descriptor, size)

    monkeypatch.setattr(os, "read", grow_before_read)

    with pytest.raises(ToolError, match="larger than 3 bytes"):
        filesystem.read_text("notes.txt", max_bytes=3)


def test_replace_failure_preserves_original_and_removes_temp(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = project / "notes.txt"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source: str, destination: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
        assert src_dir_fd == dst_dir_fd
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(ToolError, match="injected replace failure"):
        filesystem.write_text("notes.txt", "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(item.name for item in project.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize("operation", ["write", "edit", "read", "remove"])
def test_closes_descriptors_after_success(
    project: Path, filesystem: SafeProjectFS, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    directory = project / "nested"
    directory.mkdir()
    (directory / "notes.txt").write_text("old", encoding="utf-8")
    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    closed: list[int] = []

    def track_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def track_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", track_close)

    run_operation(filesystem, operation, "nested/notes.txt")

    assert len(opened) >= 2
    assert sorted(opened) == sorted(closed)


def test_zero_byte_limit_and_negative_limit(project: Path, filesystem: SafeProjectFS) -> None:
    (project / "empty.txt").touch()

    assert filesystem.read_text("empty.txt", max_bytes=0) == ""
    with pytest.raises(ToolError, match="non-negative"):
        filesystem.read_text("empty.txt", max_bytes=-1)

from __future__ import annotations

import ctypes
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent.safe_fs as safe_fs_module
import agent.tools.patch_tools as patch_tools
from agent.safe_fs import (
    MissingFileError,
    PortableProjectFS,
    ProjectFS,
    SafeProjectFS,
    project_fs,
)
from agent.security import PathSecurityError
from agent.tools.base import ToolContext, ToolError
from agent.tools.patch_tools import ApplyPatchTool


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture()
def filesystem(project: Path) -> PortableProjectFS:
    return PortableProjectFS(project)


def _case_insensitive(root: Path) -> bool:
    probe = root / "Case.Probe"
    probe.write_text("x", encoding="utf-8")
    try:
        return (root / "case.probe").exists()
    finally:
        probe.unlink()


def _require_symlinks(tmp_path: Path) -> None:
    """Skip the test where the system does not let this user create a symlink."""
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    probe.unlink()


class _FakeKernel32:
    """The two Win32 calls of the root check, with answers set per test."""

    def __init__(
        self, *, drive_type: int = 3, file_system: str = "NTFS", known: bool = True
    ) -> None:
        self.drive_type = drive_type
        self.file_system = file_system
        self.known = known
        self.drive_roots: list[str] = []

    def GetDriveTypeW(self, root: str) -> int:
        self.drive_roots.append(root)
        return self.drive_type

    def GetVolumeInformationW(
        self,
        root: str,
        _name: object,
        _name_size: int,
        _serial: object,
        _component: object,
        _flags: object,
        file_system: Any,
        _size: int,
    ) -> int:
        file_system.value = self.file_system
        return int(self.known)


def _fake_windows(
    monkeypatch: pytest.MonkeyPatch, kernel32: _FakeKernel32 | None = None
) -> _FakeKernel32:
    """Give the root check a Win32 facade; the host path rules stay in place."""
    fake = _FakeKernel32() if kernel32 is None else kernel32
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=fake), raising=False)
    return fake


def _fake_windows_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the factory take the Windows branch on any host."""
    _fake_windows(monkeypatch)
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "abspath", lambda value: value)


@pytest.mark.parametrize(
    "path",
    ["CON", "con.txt", "CON .txt", "NUL", "PRN", "AUX", "COM1", "com9.log", "com³", "lpt².txt",
     "conin$", "conout$", "dir/CON.md"],
)
def test_refuses_reserved_device_names(filesystem: ProjectFS, path: str) -> None:
    with pytest.raises(PathSecurityError, match="Reserved device names"):
        filesystem.write_text(path, "x")


@pytest.mark.parametrize("path", ["C:name", "file:stream", "dir/a:b"])
def test_refuses_drive_relative_and_stream_forms(filesystem: ProjectFS, path: str) -> None:
    with pytest.raises(PathSecurityError, match="Drive-relative paths and ADS"):
        filesystem.write_text(path, "x")


def test_refuses_a_root_relative_path(filesystem: ProjectFS) -> None:
    with pytest.raises(PathSecurityError, match="Paths from the drive root"):
        filesystem.write_text(r"\name", "x")


@pytest.mark.parametrize(
    "path", ["D:\\data", r"\\server\share\file", r"\\.\pipe\x", r"\\?\C:\x"]
)
def test_refuses_other_drives_shares_and_namespaces(
    project: Path, filesystem: ProjectFS, path: str
) -> None:
    """Another drive, a share or a device namespace never reaches the file system.

    Windows reads these as absolute paths outside the project; other systems refuse
    them in the name checks. Either way nothing is created.
    """
    with pytest.raises(PathSecurityError):
        filesystem.write_text(path, "x")
    assert list(project.iterdir()) == []


@pytest.mark.parametrize("path", ["name.", "name ..", "name\t", "dir/x.", "a...b."])
def test_refuses_trailing_dots_and_spaces(filesystem: ProjectFS, path: str) -> None:
    with pytest.raises(PathSecurityError, match="Trailing dots and spaces"):
        filesystem.write_text(path, "x")


@pytest.mark.parametrize("path", ["SECRET~1", "progra~1.txt", "dir/data~2.csv"])
def test_refuses_short_name_tails(filesystem: ProjectFS, path: str) -> None:
    with pytest.raises(PathSecurityError, match="Short 8.3"):
        filesystem.write_text(path, "x")


@pytest.mark.parametrize("path", [".env", ".env.production", "id_rsa", "dir/backup.p12"])
def test_refuses_sensitive_paths(filesystem: ProjectFS, path: str) -> None:
    with pytest.raises(PathSecurityError, match="secret-bearing"):
        filesystem.write_text(path, "x")


def test_refuses_parent_traversal_and_absolute_escape(
    filesystem: ProjectFS, tmp_path: Path
) -> None:
    with pytest.raises(PathSecurityError, match="Parent traversal"):
        filesystem.write_text("../escape.txt", "x")
    with pytest.raises(PathSecurityError, match="outside the selected project"):
        filesystem.write_text(str(tmp_path / "outside.txt"), "x")


def test_refuses_overlong_paths(filesystem: ProjectFS) -> None:
    with pytest.raises(ToolError, match="too long"):
        filesystem.write_text("a" * 240, "x")


def test_path_budget_leaves_room_for_the_temporary_file(
    filesystem: PortableProjectFS,
) -> None:
    """A target at the limit still gets its temporary file inside MAX_PATH."""
    limit = safe_fs_module._MAX_FULL_PATH_UNITS
    room = limit - safe_fs_module._utf16_units(str(filesystem.root)) - 1
    assert 40 < room < 200  # a temporary directory must leave a usable part of the budget
    at_limit = "a" * room

    filesystem.write_text(at_limit, "x")

    assert filesystem.read_text(at_limit, max_bytes=10) == "x"
    with pytest.raises(ToolError, match="too long"):
        filesystem.write_text(at_limit + "b", "x")
    written = limit + 1 + len(safe_fs_module._temp_name())
    assert written <= safe_fs_module._MAX_PATH_UNITS


def test_write_read_roundtrip_keeps_bytes_exact(project: Path, filesystem: ProjectFS) -> None:
    content = "line\r\nnext\x1a λx tail"
    relative = filesystem.write_text("doc.txt", content)
    assert str(relative) == "doc.txt"
    assert filesystem.read_text("doc.txt", max_bytes=1000) == content
    assert (project / "doc.txt").read_bytes() == content.encode("utf-8")
    assert [item.name for item in project.iterdir()] == ["doc.txt"]


def test_write_creates_nested_directories(project: Path, filesystem: ProjectFS) -> None:
    filesystem.write_text("deep/dir/doc.txt", "x")
    assert (project / "deep/dir/doc.txt").read_text(encoding="utf-8") == "x"
    with pytest.raises(ToolError, match="File operation failed"):
        filesystem.read_text("missing/dir/doc.txt", max_bytes=10)


def test_write_refuses_symlink_file(project: Path, filesystem: ProjectFS, tmp_path: Path) -> None:
    _require_symlinks(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")
    (project / "link.txt").symlink_to(outside)
    with pytest.raises(PathSecurityError, match="Symlink files"):
        filesystem.write_text("link.txt", "x")
    with pytest.raises(PathSecurityError, match="Symlink files"):
        filesystem.read_text("link.txt", max_bytes=10)
    with pytest.raises(PathSecurityError, match="Symlink files"):
        filesystem.remove_file("link.txt")
    assert outside.read_text(encoding="utf-8") == "TOP SECRET"
    assert [item.name for item in project.iterdir()] == ["link.txt"]


def test_write_refuses_dangling_symlink_escape(
    project: Path, filesystem: ProjectFS, tmp_path: Path
) -> None:
    _require_symlinks(tmp_path)
    (project / "link.txt").symlink_to(tmp_path / "missing.txt")
    with pytest.raises(PathSecurityError, match="Symlink files"):
        filesystem.write_text("link.txt", "x")
    assert not (tmp_path / "missing.txt").exists()
    assert [item.name for item in project.iterdir()] == ["link.txt"]


def test_refuses_symlink_directory_components(
    project: Path, filesystem: ProjectFS, tmp_path: Path
) -> None:
    _require_symlinks(tmp_path)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "file.txt").write_text("TOP SECRET", encoding="utf-8")
    (project / "sub").symlink_to(tmp_path / "outside")
    for operation in ("read", "write", "remove"):
        with pytest.raises(PathSecurityError, match="Symlink components"):
            if operation == "read":
                filesystem.read_text("sub/file.txt", max_bytes=10)
            elif operation == "write":
                filesystem.write_text("sub/file.txt", "x")
            else:
                filesystem.remove_file("sub/file.txt")
    assert (tmp_path / "outside" / "file.txt").read_text(encoding="utf-8") == "TOP SECRET"


def test_edit_reports_fragment_errors(filesystem: ProjectFS) -> None:
    filesystem.write_text("doc.txt", "one two one")
    with pytest.raises(ToolError, match="old_text was not found"):
        filesystem.edit_text("doc.txt", "three", "x", replace_all=False)
    with pytest.raises(ToolError, match="occurs 2 times"):
        filesystem.edit_text("doc.txt", "one", "x", replace_all=False)
    relative, count = filesystem.edit_text("doc.txt", "one", "zero", replace_all=True)
    assert (str(relative), count) == ("doc.txt", 2)
    assert filesystem.read_text("doc.txt", max_bytes=100) == "zero two zero"


def test_edit_many_rejects_oversized_result(filesystem: ProjectFS) -> None:
    filesystem.write_text("doc.txt", "x")
    with pytest.raises(ToolError, match="result would exceed"):
        filesystem.edit_many("doc.txt", [("x", "y" * 1_000_001, False)])


def test_edit_detects_conflict_before_replace(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = project / "doc.txt"
    target.write_text("original", encoding="utf-8")
    real_fsync = os.fsync
    seen: list[int] = []

    def fsync_then_tamper(descriptor: int) -> None:
        if not seen:
            seen.append(descriptor)
            target.write_text("tampered-longer", encoding="utf-8")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fsync_then_tamper)
    with pytest.raises(ToolError, match="conflict"):
        filesystem.edit_text("doc.txt", "original", "new", replace_all=False)
    assert target.read_text(encoding="utf-8") == "tampered-longer"
    assert [item.name for item in project.iterdir()] == ["doc.txt"]


def test_create_text_exclusive_publishes_complete_file(
    project: Path, filesystem: ProjectFS
) -> None:
    relative = filesystem.create_text_exclusive("skills/notes/SKILL.md", "body")
    assert str(relative) == str(Path("skills/notes/SKILL.md"))
    assert (project / "skills/notes/SKILL.md").read_text(encoding="utf-8") == "body"
    assert [item.name for item in project.iterdir()] == ["skills"]
    with pytest.raises(ToolError, match="File already exists"):
        filesystem.create_text_exclusive("skills/notes/SKILL.md", "other")


def test_create_text_exclusive_cleans_temporary_file_on_failure(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_rename(source: Path, target: Path) -> None:
        raise OSError("rename refused")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(ToolError, match="File operation failed"):
        filesystem.create_text_exclusive("doc.txt", "body")
    assert [item.name for item in project.iterdir()] == []


def test_create_text_exclusive_reports_lost_race(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    def race_rename(source: Path, target: Path) -> None:
        raise FileExistsError(17, "File exists")

    monkeypatch.setattr(os, "rename", race_rename)
    with pytest.raises(ToolError, match="File already exists"):
        filesystem.create_text_exclusive("doc.txt", "body")
    assert [item.name for item in project.iterdir()] == []


def test_remove_file_and_missing_file(project: Path, filesystem: ProjectFS) -> None:
    (project / "doc.txt").write_text("x", encoding="utf-8")
    assert str(filesystem.remove_file("doc.txt")) == "doc.txt"
    with pytest.raises(MissingFileError, match="does not exist"):
        filesystem.remove_file("doc.txt")


def test_remove_if_unchanged_refuses_stale_content(filesystem: ProjectFS) -> None:
    filesystem.write_text("doc.txt", "current")
    with pytest.raises(ToolError, match="File changed"):
        filesystem.remove_if_unchanged("doc.txt", "stale")
    assert filesystem.exists("doc.txt")


def test_remove_if_unchanged_detects_concurrent_swap(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = project / "doc.txt"
    target.write_text("original", encoding="utf-8")
    real_final = safe_fs_module._final_entry
    calls: list[Path] = []

    def final_then_tamper(path: Path) -> object:
        result = real_final(path)
        calls.append(path)
        if len(calls) == 2:  # the recheck directly before the unlink
            target.write_text("tampered-longer", encoding="utf-8")
            result = real_final(path)
        return result

    monkeypatch.setattr(safe_fs_module, "_final_entry", final_then_tamper)
    with pytest.raises(ToolError, match="File changed during delete"):
        filesystem.remove_if_unchanged("doc.txt", "original")
    assert target.read_text(encoding="utf-8") == "tampered-longer"


def test_replace_if_unchanged_roundtrip_and_stale(filesystem: ProjectFS) -> None:
    filesystem.write_text("doc.txt", "old text")
    filesystem.replace_if_unchanged("doc.txt", "old text", "new text")
    assert filesystem.read_text("doc.txt", max_bytes=100) == "new text"
    with pytest.raises(ToolError, match="File changed"):
        filesystem.replace_if_unchanged("doc.txt", "old text", "another")


def test_exists_reports_and_propagates_security_errors(filesystem: ProjectFS) -> None:
    filesystem.write_text("doc.txt", "x")
    assert filesystem.exists("doc.txt")
    assert not filesystem.exists("missing.txt")
    with pytest.raises(PathSecurityError, match="secret-bearing"):
        filesystem.exists(".env")


def test_read_limits_and_encoding(project: Path, filesystem: ProjectFS) -> None:
    filesystem.write_text("doc.txt", "abcdef")
    with pytest.raises(ToolError, match="larger than 5"):
        filesystem.read_text("doc.txt", max_bytes=5)
    with pytest.raises(ToolError, match="non-negative"):
        filesystem.read_text("doc.txt", max_bytes=-1)
    (project / "bin.dat").write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(ToolError, match="not valid UTF-8"):
        filesystem.read_text("bin.dat", max_bytes=10)
    (project / "adir").mkdir()
    with pytest.raises(ToolError, match="Only regular files"):
        filesystem.read_text("adir", max_bytes=10)


def test_case_variant_resolves_to_real_entry(project: Path, filesystem: ProjectFS) -> None:
    if not _case_insensitive(project):
        pytest.skip("case-insensitive directory entries required")
    (project / "Readme.txt").write_text("x", encoding="utf-8")
    filesystem.write_text("readme.txt", "y")
    assert [item.name for item in project.iterdir()] == ["Readme.txt"]
    assert filesystem.read_text("README.TXT", max_bytes=10) == "y"


def test_alias_name_without_real_entry_is_refused(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "real.txt").write_text("x", encoding="utf-8")
    real_listdir = os.listdir

    def only_other(path: str) -> list[str]:
        if Path(path) == project:
            return ["other.txt"]
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", only_other)
    with pytest.raises(PathSecurityError, match="does not match its real directory name"):
        filesystem.read_text("real.txt", max_bytes=10)


def test_case_ambiguous_component_is_refused(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _case_insensitive(project):
        pytest.skip("case-insensitive directory entries required")
    (project / "Doc.txt").write_text("x", encoding="utf-8")
    real_listdir = os.listdir

    def two_variants(path: str) -> list[str]:
        if Path(path) == project:
            return ["DOC.txt", "Doc.txt"]
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", two_variants)
    with pytest.raises(PathSecurityError, match="ambiguous"):
        filesystem.read_text("doc.txt", max_bytes=10)


def test_reparse_entries_are_refused_by_the_entry_checks() -> None:
    junction_dir = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
    with pytest.raises(PathSecurityError, match="Reparse point components"):
        safe_fs_module._check_directory_entry(junction_dir)
    plain_dir = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0)
    safe_fs_module._check_directory_entry(plain_dir)
    assert safe_fs_module._is_reparse(SimpleNamespace(st_file_attributes=0x400)) is True
    assert safe_fs_module._is_reparse(SimpleNamespace(st_file_attributes=0)) is False


def test_conditional_operations_fail_closed_without_file_identity() -> None:
    faceless = SimpleNamespace(st_ino=0, st_dev=0)
    with pytest.raises(ToolError, match="cannot be confirmed"):
        PortableProjectFS._require_identity(faceless)
    identified = SimpleNamespace(st_ino=12345, st_dev=678)
    PortableProjectFS._require_identity(identified)


def test_factory_prefers_the_descriptor_layer_on_posix(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX runtime required")
    assert isinstance(project_fs(project), SafeProjectFS)
    monkeypatch.setattr(os, "supports_dir_fd", frozenset())
    with pytest.raises(ToolError, match="No safe file layer"):
        project_fs(project)


def test_factory_selects_the_portable_layer_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_windows_runtime(monkeypatch)
    assert isinstance(project_fs(Path("C:/work/project")), PortableProjectFS)


def test_factory_refuses_a_root_outside_fixed_ntfs(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows the factory reports the root refusal instead of picking a layer."""
    _fake_windows_runtime(monkeypatch)
    with pytest.raises(ToolError, match="local drive letter"):
        project_fs(project)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=_FakeKernel32(drive_type=4)))
    with pytest.raises(ToolError, match="fixed local drive"):
        project_fs(Path("D:/work/project"))


def test_factory_refuses_other_platforms(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "ce")
    with pytest.raises(ToolError, match="No safe file layer"):
        project_fs(project)


@pytest.mark.parametrize(
    "root", ["C:/Users/me/project", "C:\\Users\\me\\project", "C:/", "C:\\"]
)
def test_windows_root_check_accepts_a_fixed_ntfs_drive(
    monkeypatch: pytest.MonkeyPatch, root: str
) -> None:
    kernel32 = _fake_windows(monkeypatch)

    safe_fs_module._check_windows_root(Path(root))

    assert kernel32.drive_roots == ["C:\\"]


@pytest.mark.parametrize(
    ("drive_type", "file_system", "known", "message"),
    [
        (4, "NTFS", True, "fixed local drive"),  # DRIVE_REMOTE: a mapped network drive
        (2, "NTFS", True, "fixed local drive"),  # DRIVE_REMOVABLE
        (3, "FAT32", True, "NTFS volume"),
        (3, "exFAT", True, "NTFS volume"),
        (3, "NTFS", False, "NTFS volume"),  # the volume query itself failed
    ],
)
def test_windows_root_check_refuses_unsupported_volumes(
    monkeypatch: pytest.MonkeyPatch,
    drive_type: int,
    file_system: str,
    known: bool,
    message: str,
) -> None:
    _fake_windows(monkeypatch, _FakeKernel32(drive_type=drive_type, file_system=file_system,
        known=known))

    with pytest.raises(ToolError, match=message):
        safe_fs_module._check_windows_root(Path("D:/work/project"))


@pytest.mark.parametrize(
    "root",
    [
        "/home/me/project",
        "relative/project",
        r"\\server\share\project",
        r"\\?\C:\project",
        r"\\.\C:\project",
    ],
)
def test_windows_root_check_refuses_roots_without_a_local_drive_letter(
    monkeypatch: pytest.MonkeyPatch, root: str
) -> None:
    _fake_windows(monkeypatch)

    with pytest.raises(ToolError, match="local drive letter"):
        safe_fs_module._check_windows_root(Path(root))


def test_windows_root_check_refuses_a_runtime_without_win32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ctypes, "windll", None, raising=False)

    with pytest.raises(ToolError, match="requires Windows"):
        safe_fs_module._check_windows_root(Path("C:/project"))


def test_edit_keeps_the_target_when_the_replacement_is_refused(
    project: Path, filesystem: ProjectFS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open target can make Windows refuse the replacement: nothing may change."""
    filesystem.write_text("doc.txt", "original")
    before = (project / "doc.txt").stat()

    def refuse(source: Path, target: Path) -> None:
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(ToolError, match="File operation failed"):
        filesystem.edit_text("doc.txt", "original", "changed", replace_all=False)

    assert filesystem.read_text("doc.txt", max_bytes=100) == "original"
    after = (project / "doc.txt").stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert [item.name for item in project.iterdir()] == ["doc.txt"]


async def test_apply_patch_rolls_back_through_the_portable_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure on the second file restores the first through the portable layer."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "first.txt").write_text("one\n", encoding="utf-8")
    (project / "second.txt").write_text("two\n", encoding="utf-8")
    patch = (
        "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
        "--- a/second.txt\n+++ b/second.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
    )
    real_replace = PortableProjectFS.replace_if_unchanged
    calls = {"count": 0}

    def fail_on_second(
        self: PortableProjectFS, path: str, expected: str, new: str
    ) -> Path:
        calls["count"] += 1
        if calls["count"] == 2:
            raise ToolError("injected write failure")
        return real_replace(self, path, expected, new)

    monkeypatch.setattr(PortableProjectFS, "replace_if_unchanged", fail_on_second)
    monkeypatch.setattr(patch_tools, "_safe_fs", lambda root: PortableProjectFS(root))

    with pytest.raises(ToolError, match="injected write failure"):
        await ApplyPatchTool().execute(
            ToolContext(session_id="test", project_root=project), {"patch": patch}
        )

    assert (project / "first.txt").read_text(encoding="utf-8") == "one\n"
    assert (project / "second.txt").read_text(encoding="utf-8") == "two\n"
    assert sorted(item.name for item in project.iterdir()) == ["first.txt", "second.txt"]

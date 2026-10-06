from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from agent.security import PathSecurityError, is_sensitive_path
from agent.tools.base import ToolError

_MAX_EDIT_BYTES = 1_000_000
_READ_CHUNK_BYTES = 65_536


class SafeProjectFS:
    """Perform POSIX file operations without following project symlinks."""

    def __init__(self, root: Path) -> None:
        required = {os.open, os.stat, os.mkdir, os.rename, os.unlink}
        flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if (
            os.name != "posix"
            or not required.issubset(os.supports_dir_fd)
            or os.stat not in os.supports_follow_symlinks
            or not all(hasattr(os, flag) for flag in flags)
            or not all(hasattr(os, primitive) for primitive in ("fchmod", "fsync", "replace"))
        ):
            raise ToolError("Safe file operations require POSIX dir_fd and no-follow support")
        self.root = Path(os.path.abspath(root.expanduser()))

    def write_text(self, path: str, content: str, *, create_parents: bool = True) -> Path:
        relative = self._relative_path(path)
        with self._parent(relative, create_parents=create_parents) as (parent_fd, name):
            existing = self._file_stat(name, parent_fd)
            self._replace_text(name, parent_fd, content, existing, check_conflict=False)
        return relative

    def edit_text(self, path: str, old: str, new: str, *, replace_all: bool) -> tuple[Path, int]:
        relative = self._relative_path(path)
        with self._parent(relative) as (parent_fd, name):
            content, existing = self._read_regular(name, parent_fd, _MAX_EDIT_BYTES)
            if not old or old not in content:
                raise ToolError("old_text was not found")
            count = content.count(old)
            if count > 1 and not replace_all:
                raise ToolError(f"old_text occurs {count} times; set replace_all=true")
            updated = content.replace(old, new, -1 if replace_all else 1)
            self._replace_text(name, parent_fd, updated, existing, check_conflict=True)
        return relative, count if replace_all else 1

    def edit_many(self, path: str, edits: Sequence[tuple[str, str, bool]]) -> tuple[Path, int]:
        relative = self._relative_path(path)
        with self._parent(relative) as (parent_fd, name):
            content, existing = self._read_regular(name, parent_fd, _MAX_EDIT_BYTES)
            content_bytes = len(content.encode("utf-8"))
            replacements = 0
            for number, (old, new, replace_all) in enumerate(edits, 1):
                if not old or old not in content:
                    raise ToolError(f"Edit {number}: old_text was not found")
                count = content.count(old)
                if count > 1 and not replace_all:
                    raise ToolError(
                        f"Edit {number}: old_text occurs {count} times; set replace_all=true"
                    )
                occurrence_count = count if replace_all else 1
                try:
                    delta_bytes = occurrence_count * (
                        len(new.encode("utf-8")) - len(old.encode("utf-8"))
                    )
                except UnicodeEncodeError as exc:
                    raise ToolError(f"Edit {number}: text is not valid UTF-8") from exc
                if content_bytes + delta_bytes > _MAX_EDIT_BYTES:
                    raise ToolError(f"Edit {number}: result would exceed {_MAX_EDIT_BYTES} bytes")
                content = content.replace(old, new, -1 if replace_all else 1)
                content_bytes += delta_bytes
                replacements += occurrence_count
            self._replace_text(name, parent_fd, content, existing, check_conflict=True)
        return relative, replacements

    def read_text(self, path: str, *, max_bytes: int) -> str:
        relative = self._relative_path(path)
        if max_bytes < 0:
            raise ToolError("max_bytes must be non-negative")
        with self._parent(relative) as (parent_fd, name):
            content, _existing = self._read_regular(name, parent_fd, max_bytes)
        return content

    def remove_file(self, path: str) -> Path:
        relative = self._relative_path(path)
        with self._parent(relative) as (parent_fd, name):
            if self._file_stat(name, parent_fd) is None:
                raise ToolError("File does not exist")
            os.unlink(name, dir_fd=parent_fd)
            self._sync_directory(parent_fd)
        return relative

    def _relative_path(self, value: str) -> Path:
        if "\0" in value:
            raise PathSecurityError("NUL is not allowed in file paths")
        raw = Path(value).expanduser()
        if ".." in raw.parts:
            raise PathSecurityError("Parent traversal is not allowed")
        if raw.is_absolute():
            try:
                relative = raw.relative_to(self.root)
            except ValueError as exc:
                raise PathSecurityError("Path is outside the selected project") from exc
        else:
            relative = raw
        if not relative.parts:
            raise ToolError("Path must name a file")
        if is_sensitive_path(relative):
            raise PathSecurityError("Access to secret-bearing files is blocked")
        return relative

    @contextmanager
    def _parent(self, relative: Path, *, create_parents: bool = False) -> Iterator[tuple[int, str]]:
        descriptors: list[int] = []
        try:
            parent_fd = self._open_directory(str(self.root), None)
            descriptors.append(parent_fd)
            for component in relative.parts[:-1]:
                try:
                    descriptor = self._open_directory(component, parent_fd)
                except FileNotFoundError:
                    if not create_parents:
                        raise
                    with suppress(FileExistsError):
                        os.mkdir(component, dir_fd=parent_fd)
                    descriptor = self._open_directory(component, parent_fd)
                descriptors.append(descriptor)
                parent_fd = descriptor
            yield parent_fd, relative.name
        except OSError as exc:
            raise ToolError(f"File operation failed: {exc}") from exc
        finally:
            for descriptor in reversed(descriptors):
                with suppress(OSError):
                    os.close(descriptor)

    @staticmethod
    def _entry_stat(name: str, parent_fd: int | None) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def _open_directory(self, name: str, parent_fd: int | None) -> int:
        try:
            return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        except OSError:
            existing = self._entry_stat(name, parent_fd)
            if existing is not None and stat.S_ISLNK(existing.st_mode):
                raise PathSecurityError("Symlink components are not allowed") from None
            raise

    def _file_stat(self, name: str, parent_fd: int) -> os.stat_result | None:
        existing = self._entry_stat(name, parent_fd)
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise PathSecurityError("Symlink files are not allowed")
            if not stat.S_ISREG(existing.st_mode):
                raise ToolError("Only regular files are allowed")
        return existing

    @staticmethod
    def _same_version(first: os.stat_result, second: os.stat_result) -> bool:
        return (
            first.st_dev,
            first.st_ino,
            first.st_size,
            first.st_mtime_ns,
            first.st_ctime_ns,
        ) == (
            second.st_dev,
            second.st_ino,
            second.st_size,
            second.st_mtime_ns,
            second.st_ctime_ns,
        )

    def _read_regular(
        self, name: str, parent_fd: int, max_bytes: int
    ) -> tuple[str, os.stat_result]:
        initial = self._file_stat(name, parent_fd)
        if initial is None:
            raise ToolError("File does not exist")
        try:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
        except OSError:
            self._file_stat(name, parent_fd)
            raise
        try:
            existing = os.fstat(descriptor)
            if not stat.S_ISREG(existing.st_mode):
                raise ToolError("Only regular files are allowed")
            if not self._same_version(initial, existing):
                raise ToolError("File changed during read (conflict)")
            if existing.st_size > max_bytes:
                raise ToolError(f"File is larger than {max_bytes} bytes")
            data = bytearray()
            while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes - len(data) + 1)):
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ToolError(f"File is larger than {max_bytes} bytes")
            if not self._same_version(existing, os.fstat(descriptor)):
                raise ToolError("File changed during read (conflict)")
            try:
                return data.decode("utf-8"), existing
            except UnicodeDecodeError as exc:
                raise ToolError("File is not valid UTF-8 text") from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _create_temp(parent_fd: int) -> tuple[str, int]:
        for _attempt in range(10):
            name = f".safe-fs-{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            return name, descriptor
        raise ToolError("Could not create an exclusive temporary file")

    def _replace_text(
        self,
        name: str,
        parent_fd: int,
        content: str,
        existing: os.stat_result | None,
        *,
        check_conflict: bool,
    ) -> None:
        try:
            data = memoryview(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ToolError("Content is not valid UTF-8 text") from exc
        temporary, descriptor = self._create_temp(parent_fd)
        try:
            while data:
                written = os.write(descriptor, data)
                if written <= 0:
                    raise ToolError("Could not write temporary file")
                data = data[written:]
            if existing is not None:
                os.fchmod(descriptor, stat.S_IMODE(existing.st_mode) & 0o777)
            os.fsync(descriptor)
            current = self._file_stat(name, parent_fd)
            if check_conflict and (
                existing is None or current is None or not self._same_version(existing, current)
            ):
                raise ToolError("File changed during edit (conflict)")
            os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temporary = ""
            self._sync_directory(parent_fd)
        finally:
            try:
                os.close(descriptor)
            finally:
                if temporary:
                    with suppress(OSError):
                        os.unlink(temporary, dir_fd=parent_fd)

    @staticmethod
    def _sync_directory(parent_fd: int) -> None:
        with suppress(OSError):
            os.fsync(parent_fd)

"""Safe file operations for the project directory.

Two layers implement one contract (:class:`ProjectFS`). The descriptor layer
(``SafeProjectFS``) runs on POSIX with ``dir_fd`` and ``O_NOFOLLOW`` support: it opens
every path component without following project symlinks, writes through temporary files
and replaces atomically. The portable layer (``PortableProjectFS``) runs on Windows with
the same checks on other primitives: components are verified with ``lstat`` against
symlinks and reparse points and against their real directory names, writes go through a
temporary file and an atomic ``os.replace``. The factory :func:`project_fs` selects the
layer for the platform and refuses to run where neither layer is supported.
"""

from __future__ import annotations

import ntpath
import os
import re
import secrets
import stat
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from agent.security import PathSecurityError, is_sensitive_path
from agent.tools.base import ToolError

_MAX_EDIT_BYTES = 1_000_000
_READ_CHUNK_BYTES = 65_536


def _posix_primitives_available() -> bool:
    """Return True when this runtime gives every primitive of the descriptor layer."""
    required = {os.open, os.stat, os.mkdir, os.rename, os.unlink, os.link}
    flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    return (
        os.name == "posix"
        and required.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and os.link in os.supports_follow_symlinks
        and all(hasattr(os, flag) for flag in flags)
        and all(hasattr(os, primitive) for primitive in ("fchmod", "fsync", "replace"))
    )


class MissingFileError(ToolError):
    """The file does not exist."""


class ProjectFS(ABC):
    """Contract and shared path validation of the project file layers."""

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(root.expanduser()))

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

    @abstractmethod
    def write_text(self, path: str, content: str, *, create_parents: bool = True) -> Path:
        """Create or overwrite a UTF-8 text file; returns the project-relative path."""
        raise NotImplementedError

    @abstractmethod
    def create_text_exclusive(self, path: str, content: str) -> Path:
        """Create a file only when the name is free; the file is published complete."""
        raise NotImplementedError

    @abstractmethod
    def edit_text(self, path: str, old: str, new: str, *, replace_all: bool) -> tuple[Path, int]:
        """Replace text in a file; returns the relative path and the replacement count."""
        raise NotImplementedError

    @abstractmethod
    def edit_many(self, path: str, edits: Sequence[tuple[str, str, bool]]) -> tuple[Path, int]:
        """Apply ordered text edits to one file; returns the path and the total count."""
        raise NotImplementedError

    @abstractmethod
    def read_text(self, path: str, *, max_bytes: int) -> str:
        """Read a UTF-8 text file of at most ``max_bytes`` bytes."""
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, path: str) -> Path:
        """Remove one regular file."""
        raise NotImplementedError

    @abstractmethod
    def remove_if_unchanged(self, path: str, expected: str) -> Path:
        """Remove the file only when its content still equals ``expected``."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return whether the file exists without following unsafe entries."""
        raise NotImplementedError

    @abstractmethod
    def replace_if_unchanged(self, path: str, expected: str, new: str) -> Path:
        """Replace the file content only when it still equals ``expected``."""
        raise NotImplementedError


class SafeProjectFS(ProjectFS):
    """Descriptor layer: POSIX file operations without following project symlinks."""

    def __init__(self, root: Path) -> None:
        if not _posix_primitives_available():
            raise ToolError("Safe file operations require POSIX dir_fd and no-follow support")
        super().__init__(root)

    def write_text(self, path: str, content: str, *, create_parents: bool = True) -> Path:
        relative = self._relative_path(path)
        with self._parent(relative, create_parents=create_parents) as (parent_fd, name):
            existing = self._file_stat(name, parent_fd)
            self._replace_text(name, parent_fd, content, existing, check_conflict=False)
        return relative

    def create_text_exclusive(self, path: str, content: str) -> Path:
        """Create a file only if it does not already exist.

        The creation is atomic and never follows a symlink. It writes the
        content to a unique temporary file and links it into place with
        ``os.link``, which fails with ``FileExistsError`` when the target
        exists. Raises :class:`ToolError` ("File already exists") in that case.
        """
        relative = self._relative_path(path)
        with self._parent(relative, create_parents=True) as (parent_fd, name):
            if self._file_stat(name, parent_fd) is not None:
                raise ToolError("File already exists")
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
                os.fsync(descriptor)
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise ToolError("File already exists") from exc
                # os.link adds a second name; remove the temporary name, or it stays behind.
                os.unlink(temporary, dir_fd=parent_fd)
                temporary = ""
                self._sync_directory(parent_fd)
            finally:
                try:
                    os.close(descriptor)
                finally:
                    if temporary:
                        with suppress(OSError):
                            os.unlink(temporary, dir_fd=parent_fd)
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
                raise MissingFileError("File does not exist")
            os.unlink(name, dir_fd=parent_fd)
            self._sync_directory(parent_fd)
        return relative

    def remove_if_unchanged(self, path: str, expected: str) -> Path:
        """Remove the file only if its content equals ``expected``.

        Reads the file and compares it to ``expected``. Rechecks the file
        before the unlink to guard against a concurrent change between the
        read and the unlink. Raises :class:`ToolError` ("File changed
        (conflict)") on a mismatch.

        Limitation: the unlink itself is not a compare-and-swap. A concurrent
        replacement between the final version check and the unlink could
        remove the replacement file. This narrow window is inherent to unlink.
        """
        relative = self._relative_path(path)
        with self._parent(relative) as (parent_fd, name):
            content, existing = self._read_regular(name, parent_fd, _MAX_EDIT_BYTES)
            if content != expected:
                raise ToolError("File changed (conflict)")
            current = self._file_stat(name, parent_fd)
            if current is None or not self._same_version(existing, current):
                raise ToolError("File changed during delete (conflict)")
            os.unlink(name, dir_fd=parent_fd)
            self._sync_directory(parent_fd)
        return relative

    def exists(self, path: str) -> bool:
        """Return whether the file exists without following symlinks.

        Returns ``False`` when the file or any parent directory is missing or
        cannot be safely opened.
        """
        relative = self._relative_path(path)
        try:
            with self._parent(relative) as (parent_fd, name):
                return self._file_stat(name, parent_fd) is not None
        except ToolError:
            return False

    def replace_if_unchanged(self, path: str, expected: str, new: str) -> Path:
        """Atomically replace the file content only if it equals ``expected``.

        Reads the file once and compares it to ``expected``; raises
        :class:`ToolError` ("File changed (conflict)") on a mismatch. Writes via
        :meth:`_replace_text` with ``check_conflict=True`` to guard against
        a concurrent change between the read and the replace.
        """
        relative = self._relative_path(path)
        with self._parent(relative) as (parent_fd, name):
            content, existing = self._read_regular(name, parent_fd, _MAX_EDIT_BYTES)
            if content != expected:
                raise ToolError("File changed (conflict)")
            self._replace_text(name, parent_fd, new, existing, check_conflict=True)
        return relative

    @contextmanager
    def _parent(
        self, relative: Path, *, create_parents: bool = False
    ) -> Iterator[tuple[int, str]]:
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
            raise MissingFileError("File does not exist")
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
            name = _temp_name()
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


# The portable layer rejects Windows name forms that could bypass the shared path checks
# or resolve outside the project. The reserved set covers the DOS device names, including
# the console aliases and the superscript spellings of COM1-3 and LPT1-3.
_WINDOWS_RESERVED_DEVICES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "conin$",
        "conout$",
        *(f"com{digit}" for digit in range(1, 10)),
        *(f"lpt{digit}" for digit in range(1, 10)),
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)
_SHORT_NAME_TAIL = re.compile(r"~\d")
_REPARSE_ATTRIBUTE = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
_TEMP_PREFIX = ".safe-fs-"
_TEMP_SUFFIX = ".tmp"
# A write puts the temporary file next to its target. The budget keeps both paths inside
# MAX_PATH, so a target that passes the check never fails later on its temporary name.
_TEMP_NAME_UNITS = 1 + len(_TEMP_PREFIX) + 32 + len(_TEMP_SUFFIX)  # with the separator
_MAX_PATH_UNITS = 259  # MAX_PATH without the terminating NUL
_MAX_FULL_PATH_UNITS = _MAX_PATH_UNITS - _TEMP_NAME_UNITS

_PORTABLE_READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0)
_PORTABLE_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)


def _temp_name() -> str:
    """Return the name of the temporary file of one write: 16 random bytes as hex."""
    return f"{_TEMP_PREFIX}{secrets.token_hex(16)}{_TEMP_SUFFIX}"


def _utf16_units(text: str) -> int:
    """Return the length of ``text`` in UTF-16 code units, as the platform counts paths."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _is_reparse(entry: os.stat_result) -> bool:
    """Return True when the entry carries the reparse point attribute."""
    return bool(int(getattr(entry, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _check_directory_entry(entry: os.stat_result) -> None:
    """Refuse links and non-directories in an intermediate path component."""
    if stat.S_ISLNK(entry.st_mode):
        raise PathSecurityError("Symlink components are not allowed")
    if _is_reparse(entry):
        raise PathSecurityError("Reparse point components are not allowed")
    if not stat.S_ISDIR(entry.st_mode):
        raise ToolError("Path component is not a directory")


def _final_entry(path: Path) -> os.stat_result | None:
    """Return the stat of a regular final entry, or None when it does not exist."""
    entry = _lstat(path)
    if entry is None:
        return None
    if stat.S_ISLNK(entry.st_mode):
        raise PathSecurityError("Symlink files are not allowed")
    if _is_reparse(entry):
        raise PathSecurityError("Reparse point entries are not allowed")
    if not stat.S_ISREG(entry.st_mode):
        raise ToolError("Only regular files are allowed")
    return entry


def _canonical_name(parent: Path, name: str) -> str:
    """Return the real directory-entry spelling of an existing component name.

    A requested name must either match a real entry exactly or match exactly one
    case-insensitive variant: several variants mean a case-sensitive directory where the
    request is ambiguous, and no variant at all with an existing entry means a file system
    alias (for example a short 8.3 name) that bypasses the name checks.
    """
    try:
        names = os.listdir(parent)
    except OSError as exc:
        raise ToolError(f"File operation failed: {exc}") from exc
    if name in names:
        return name
    variants = [entry for entry in names if entry.casefold() == name.casefold()]
    if len(variants) == 1:
        return variants[0]
    if variants:
        raise PathSecurityError("Path component is ambiguous in a case-sensitive directory")
    raise PathSecurityError("Path component does not match its real directory name")


class PortableProjectFS(ProjectFS):
    """Portable layer for Windows: the shared checks on non-descriptor primitives.

    Every component below the root is verified with ``lstat`` against symlinks and
    reparse points and against its real directory-entry name. Writes go through a
    temporary file in the same directory and an atomic ``os.replace``; the exclusive
    creation publishes a complete temporary file with ``os.rename``. The layer keeps
    no descriptors open across the publication, does not preserve file modes and does
    not fsync directories: the platform does not offer these operations.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        # The trusted operator configuration resolves the root once; a junction in the
        # root itself is the operator's choice, like a bind mount on POSIX.
        self.root = Path(os.path.realpath(self.root))

    def _relative_path(self, value: str) -> Path:
        relative = super()._relative_path(value)
        self._check_windows_names(value, relative)
        if _utf16_units(str(self.root / relative)) > _MAX_FULL_PATH_UNITS:
            raise ToolError("Path is too long for the portable file layer")
        return relative

    @staticmethod
    def _check_windows_names(value: str, relative: Path) -> None:
        if value.startswith("\\"):
            raise PathSecurityError("Paths from the drive root are not allowed")
        for component in relative.parts:
            if not component.strip("/\\"):
                raise PathSecurityError("Paths from the drive root are not allowed")
            if ":" in component:
                raise PathSecurityError("Drive-relative paths and ADS streams are not allowed")
            if component != component.rstrip(". \t"):
                raise PathSecurityError("Trailing dots and spaces are not allowed in file names")
            if _SHORT_NAME_TAIL.search(component):
                raise PathSecurityError("Short 8.3 name aliases are not allowed")
            stem = component.split(".", 1)[0].rstrip(" \t").casefold()
            if stem in _WINDOWS_RESERVED_DEVICES:
                raise PathSecurityError("Reserved device names are not allowed")

    def _resolve(self, relative: Path, *, create_parents: bool) -> tuple[Path, Path]:
        """Walk to the target parent, checking every existing component by name."""
        parent = self.root
        try:
            for component in relative.parts[:-1]:
                child = parent / component
                entry = _lstat(child)
                if entry is None:
                    if not create_parents:
                        raise FileNotFoundError(str(child))
                    with suppress(FileExistsError):
                        os.mkdir(child)
                    entry = _lstat(child)
                    if entry is None:
                        raise FileNotFoundError(str(child))
                _check_directory_entry(entry)
                parent = parent / _canonical_name(parent, component)
            name = relative.name
            if _lstat(parent / name) is not None:
                name = _canonical_name(parent, name)
        except OSError as exc:
            raise ToolError(f"File operation failed: {exc}") from exc
        return parent / name, parent

    @staticmethod
    def _same_version(first: os.stat_result, second: os.stat_result) -> bool:
        return (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) == (
            second.st_dev,
            second.st_ino,
            second.st_size,
            second.st_mtime_ns,
        )

    @staticmethod
    def _require_identity(entry: os.stat_result) -> None:
        """Fail closed when the platform cannot identify the file for a conditional check."""
        if not entry.st_ino or not entry.st_dev:
            raise ToolError("File version cannot be confirmed on this platform")

    def _read_regular(self, path: Path, max_bytes: int) -> tuple[str, os.stat_result]:
        initial = _final_entry(path)
        if initial is None:
            raise MissingFileError("File does not exist")
        try:
            descriptor = os.open(path, _PORTABLE_READ_FLAGS)
        except OSError as exc:
            raise ToolError(f"File operation failed: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ToolError("Only regular files are allowed")
            if not self._same_version(initial, opened):
                raise ToolError("File changed during read (conflict)")
            if opened.st_size > max_bytes:
                raise ToolError(f"File is larger than {max_bytes} bytes")
            data = bytearray()
            while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes - len(data) + 1)):
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ToolError(f"File is larger than {max_bytes} bytes")
            if not self._same_version(opened, os.fstat(descriptor)):
                raise ToolError("File changed during read (conflict)")
            try:
                return data.decode("utf-8"), opened
            except UnicodeDecodeError as exc:
                raise ToolError("File is not valid UTF-8 text") from exc
        finally:
            os.close(descriptor)

    def _replace_text(
        self,
        path: Path,
        parent: Path,
        content: str,
        existing: os.stat_result | None,
        *,
        check_conflict: bool,
    ) -> None:
        try:
            data = memoryview(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ToolError("Content is not valid UTF-8 text") from exc
        if check_conflict:
            if existing is None:
                raise ToolError("File changed during edit (conflict)")
            self._require_identity(existing)
        temporary = parent / _temp_name()
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, _PORTABLE_WRITE_FLAGS, 0o600)
            while data:
                written = os.write(descriptor, data)
                if written <= 0:
                    raise ToolError("Could not write temporary file")
                data = data[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            current = _final_entry(path)
            if check_conflict and (
                existing is None or current is None or not self._same_version(existing, current)
            ):
                raise ToolError("File changed during edit (conflict)")
            try:
                os.replace(temporary, path)
            except OSError as exc:
                raise ToolError(f"File operation failed: {exc}") from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary)

    def write_text(self, path: str, content: str, *, create_parents: bool = True) -> Path:
        relative = self._relative_path(path)
        target, parent = self._resolve(relative, create_parents=create_parents)
        existing = _final_entry(target)
        self._replace_text(target, parent, content, existing, check_conflict=False)
        return relative

    def create_text_exclusive(self, path: str, content: str) -> Path:
        """Publish a complete temporary file under a free name, without overwriting."""
        relative = self._relative_path(path)
        target, parent = self._resolve(relative, create_parents=True)
        if _final_entry(target) is not None:
            raise ToolError("File already exists")
        try:
            data = memoryview(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ToolError("Content is not valid UTF-8 text") from exc
        temporary = parent / _temp_name()
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, _PORTABLE_WRITE_FLAGS, 0o600)
            try:
                while data:
                    written = os.write(descriptor, data)
                    if written <= 0:
                        raise ToolError("Could not write temporary file")
                    data = data[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                descriptor = None
            # On Windows os.rename refuses a taken name, so a complete file is published
            # without overwriting. On POSIX this pre-check keeps the same contract.
            if _final_entry(target) is not None:
                raise ToolError("File already exists")
            try:
                os.rename(temporary, target)
            except FileExistsError as exc:
                raise ToolError("File already exists") from exc
            except OSError as exc:
                raise ToolError(f"File operation failed: {exc}") from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary)
        return relative

    def edit_text(self, path: str, old: str, new: str, *, replace_all: bool) -> tuple[Path, int]:
        relative = self._relative_path(path)
        target, parent = self._resolve(relative, create_parents=False)
        content, existing = self._read_regular(target, _MAX_EDIT_BYTES)
        if not old or old not in content:
            raise ToolError("old_text was not found")
        count = content.count(old)
        if count > 1 and not replace_all:
            raise ToolError(f"old_text occurs {count} times; set replace_all=true")
        updated = content.replace(old, new, -1 if replace_all else 1)
        self._replace_text(target, parent, updated, existing, check_conflict=True)
        return relative, count if replace_all else 1

    def edit_many(self, path: str, edits: Sequence[tuple[str, str, bool]]) -> tuple[Path, int]:
        relative = self._relative_path(path)
        target, parent = self._resolve(relative, create_parents=False)
        content, existing = self._read_regular(target, _MAX_EDIT_BYTES)
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
        self._replace_text(target, parent, content, existing, check_conflict=True)
        return relative, replacements

    def read_text(self, path: str, *, max_bytes: int) -> str:
        if max_bytes < 0:
            raise ToolError("max_bytes must be non-negative")
        relative = self._relative_path(path)
        target, _parent = self._resolve(relative, create_parents=False)
        content, _existing = self._read_regular(target, max_bytes)
        return content

    def remove_file(self, path: str) -> Path:
        relative = self._relative_path(path)
        target, _parent = self._resolve(relative, create_parents=False)
        if _final_entry(target) is None:
            raise MissingFileError("File does not exist")
        try:
            os.unlink(target)
        except OSError as exc:
            raise ToolError(f"File operation failed: {exc}") from exc
        return relative

    def remove_if_unchanged(self, path: str, expected: str) -> Path:
        relative = self._relative_path(path)
        target, _parent = self._resolve(relative, create_parents=False)
        content, existing = self._read_regular(target, _MAX_EDIT_BYTES)
        if content != expected:
            raise ToolError("File changed (conflict)")
        self._require_identity(existing)
        current = _final_entry(target)
        if current is None or not self._same_version(existing, current):
            raise ToolError("File changed during delete (conflict)")
        try:
            os.unlink(target)
        except OSError as exc:
            raise ToolError(f"File operation failed: {exc}") from exc
        return relative

    def exists(self, path: str) -> bool:
        relative = self._relative_path(path)
        try:
            target, _parent = self._resolve(relative, create_parents=False)
        except (ToolError, OSError):
            return False
        try:
            return _final_entry(target) is not None
        except ToolError:
            return False

    def replace_if_unchanged(self, path: str, expected: str, new: str) -> Path:
        relative = self._relative_path(path)
        target, parent = self._resolve(relative, create_parents=False)
        content, existing = self._read_regular(target, _MAX_EDIT_BYTES)
        if content != expected:
            raise ToolError("File changed (conflict)")
        self._replace_text(target, parent, new, existing, check_conflict=True)
        return relative


def _check_windows_root(root: Path) -> None:
    """Refuse a root that is not on a fixed NTFS drive.

    The portable layer is accepted only there: ``os.rename`` and ``os.replace`` have the
    checked semantics on local NTFS. ``ntpath`` parses the drive on every platform, so
    the check behaves the same wherever it runs.
    """
    import ctypes

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        raise ToolError("The portable file layer requires Windows")
    drive, _tail = ntpath.splitdrive(str(root))
    # Only "C:" is a local drive. A UNC share and the \\?\ or \\.\ namespaces have none.
    if not (len(drive) == 2 and drive[0].isascii() and drive[0].isalpha() and drive[1] == ":"):
        raise ToolError("The portable file layer requires a local drive letter")
    root_path = f"{drive}\\"
    if windll.kernel32.GetDriveTypeW(root_path) != 3:  # DRIVE_FIXED
        raise ToolError("The portable file layer requires a fixed local drive")
    file_system = ctypes.create_unicode_buffer(16)
    serial = ctypes.c_ulong()
    max_component = ctypes.c_ulong()
    flags = ctypes.c_ulong()
    ok = windll.kernel32.GetVolumeInformationW(
        root_path,
        None,
        0,
        ctypes.byref(serial),
        ctypes.byref(max_component),
        ctypes.byref(flags),
        file_system,
        16,
    )
    if not ok or file_system.value.upper() != "NTFS":
        raise ToolError("The portable file layer requires an NTFS volume")


def project_fs(root: Path) -> ProjectFS:
    """Return the strongest safe file layer available on this platform.

    POSIX with full descriptor support keeps the descriptor layer. Windows gets the
    portable layer for a root on a fixed NTFS drive. Any other runtime is refused by
    name: the caller sees the platform limit instead of a silently weakened layer.
    """
    if _posix_primitives_available():
        return SafeProjectFS(root)
    if os.name == "nt":
        # The check needs an absolute path: a drive-relative root has no volume to inspect.
        root = Path(os.path.abspath(root.expanduser()))
        _check_windows_root(root)
        return PortableProjectFS(root)
    raise ToolError(
        "No safe file layer is available on this platform: the descriptor layer needs "
        "POSIX dir_fd support, the portable layer needs Windows"
    )

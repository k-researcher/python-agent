"""Operating system sandbox for shell commands.

The sandbox confines a command to the project. The command can read and write the project
directory and a private temporary directory. It can read a small set of system and runtime
paths. It cannot read other user files, write outside these directories, open network
connections, inspect other processes or use shared memory.

Trust rule: only the trusted configuration and the agent process select the allowed paths.
Files in the project never change the profile, because the sandboxed code can write them.

macOS uses ``sandbox-exec`` with a deny-by-default profile. Other systems have no backend yet:
the "required" mode then refuses to run the command.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.config import Settings

SandboxMode = Literal["off", "auto", "required"]

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
XCODE_SELECT_LINK = "/private/var/db/xcode_select_link"

# Read-only system locations for the dynamic loader, the C library and the developer tools.
SYSTEM_READ_PATHS = (
    "/bin",
    "/sbin",
    "/usr",
    "/System",
    "/Library/Developer/CommandLineTools",
    "/private/var/db/timezone",
    "/private/var/db/dyld",
    "/private/var/select",  # /bin/sh starts the shell that this link selects
)

# Single system files that programs read at start.
SYSTEM_READ_FILES = (
    "/private/etc/localtime",
    "/private/etc/passwd",
    "/private/etc/group",
    "/private/etc/protocols",
    "/private/etc/services",
)

# Parts of the allowed trees that hold user data or third-party configuration.
DENIED_READ_PATHS = (
    "/usr/local",
    "/System/Volumes",
)

# Mach service for user and group lookups (getpwuid). Other services stay closed.
MACH_SERVICES = ("com.apple.system.opendirectoryd.libinfo",)

# Read-only kernel values that interpreters and build tools query.
SYSCTL_PREFIXES = ("hw.", "kern.os", "machdep.cpu.")
SYSCTL_NAMES = (
    "kern.argmax",
    "kern.boottime",
    "kern.hostname",
    "kern.maxfiles",
    "kern.maxfilesperproc",
    "kern.ngroups",
    "kern.safeboot",
    "kern.secure_kernel",
    "kern.usrstack64",
    "kern.version",
    "sysctl.proc_cputype",
    "sysctl.proc_translated",
    "vm.loadavg",
    "vm.pagesize",
)

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class SandboxUnavailableError(RuntimeError):
    """The required sandbox cannot run on this system or cannot run safely."""


def runtime_paths() -> tuple[str, ...]:
    """Return the trusted runtime directories of the agent process.

    The agent calls this before any sandboxed code runs. The values come from the running
    interpreter and from the system developer tools link, not from project files.
    """
    paths = {sys.base_prefix, os.path.realpath(sys.base_prefix)}
    if os.path.islink(XCODE_SELECT_LINK):
        paths.add(os.path.realpath(XCODE_SELECT_LINK))
    return tuple(sorted(path for path in paths if path))


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    mode: SandboxMode = "off"
    temp_root: Path | None = None  # parent of the per-session temporary directories
    extra_read_paths: tuple[str, ...] = ()  # trusted operator configuration only
    allow_network: bool = False
    runtime_read_paths: tuple[str, ...] = field(default_factory=runtime_paths)


@functools.cache
def available() -> bool:
    """Return True when the sandbox backend can apply a profile here.

    The check runs a trivial command once. A command that already runs inside a sandbox
    cannot apply another one, so the check also fails there.
    """
    if sys.platform != "darwin" or not os.access(SANDBOX_EXEC, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _quote(value: str) -> str:
    """Return a profile string literal. Control characters are refused."""
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise SandboxUnavailableError("A sandbox path contains a control character")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _both_forms(paths: Iterable[str]) -> list[str]:
    """Return each absolute path and its resolved form; the kernel matches the real path."""
    result: list[str] = []
    for path in paths:
        for form in (os.path.abspath(path), os.path.realpath(path)):
            if form not in result:
                result.append(form)
    return result


def _ancestors(paths: Iterable[str]) -> list[str]:
    """Return the parent directories of the paths, for path resolution (metadata only)."""
    result: list[str] = []
    for path in paths:
        current = Path(path)
        for parent in current.parents:
            text = str(parent)
            if text not in result:
                result.append(text)
    return result


def _filters(kind: str, values: Iterable[str]) -> str:
    return " ".join(f"({kind} {_quote(value)})" for value in values)


def profile(
    project_root: Path,
    temp_dir: Path,
    read_paths: Sequence[str],
    *,
    allow_network: bool = False,
) -> str:
    """Return a deny-by-default sandbox-exec profile for one command."""
    # The temporary directory is used exactly as checked: no new path resolution here, so a
    # link that replaces it later does not open its target.
    writable = [*_both_forms([str(project_root)]), str(temp_dir)]
    readable = _both_forms([*SYSTEM_READ_PATHS, *read_paths])
    files = _both_forms(SYSTEM_READ_FILES)
    parents = _ancestors([*writable, *readable, *files])
    sysctl = (
        _filters("sysctl-name-prefix", SYSCTL_PREFIXES)
        + " "
        + _filters("sysctl-name", SYSCTL_NAMES)
    )
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(deny process-info*)",
        "(allow process-info* (target self))",
        "(allow signal (target same-sandbox))",
        f"(allow sysctl-read {sysctl})",
        f"(allow mach-lookup {_filters('global-name', MACH_SERVICES)})",
        f"(allow file-read-metadata {_filters('literal', parents)})",
        # The dynamic loader lists the root directory at start. It shows top-level names only.
        '(allow file-read-data (literal "/"))',
        f"(allow file-read* {_filters('subpath', readable)} {_filters('literal', files)})",
        f"(allow file-read* file-write* {_filters('subpath', writable)})",
        '(allow file-read* file-write-data (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom"))',
        # Later rules win: close user data inside the allowed system trees.
        f"(deny file-read* {_filters('subpath', DENIED_READ_PATHS)})",
    ]
    if allow_network:
        lines.append("(allow network*)")
    return "\n".join(lines) + "\n"


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def session_temp_dir(temp_root: Path, session_id: str, project_root: Path) -> Path:
    """Create or check the private temporary directory of one session.

    The directory must be a real directory (not a link), owned by this user, with no access
    for others. Any other state stops the command: the sandboxed code may try to replace the
    directory with a link to user files.
    """
    if not _SESSION_ID.fullmatch(session_id):
        raise SandboxUnavailableError("The session ID is not valid for a sandbox directory")
    root = Path(os.path.realpath(temp_root))
    project = Path(os.path.realpath(project_root))
    if _inside(root, project) or _inside(project, root):
        raise SandboxUnavailableError("The sandbox temporary directory must be outside the project")
    root.mkdir(parents=True, exist_ok=True)
    root_state = os.lstat(root)
    if not stat.S_ISDIR(root_state.st_mode) or root_state.st_uid != os.getuid():
        raise SandboxUnavailableError("The sandbox temporary root is not a directory of this user")
    os.chmod(root, 0o700)
    path = root / session_id
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, 0o700)
    state = os.lstat(path)
    if (
        not stat.S_ISDIR(state.st_mode)
        or state.st_uid != os.getuid()
        or stat.S_IMODE(state.st_mode) != 0o700
    ):
        raise SandboxUnavailableError("The sandbox temporary directory was changed; stop")
    return path


def prepare(
    policy: SandboxPolicy, project_root: Path, session_id: str
) -> tuple[list[str], dict[str, str]] | None:
    """Return the command prefix and environment overrides, or None when the sandbox is off.

    Raise SandboxUnavailableError when the mode is "required" and the sandbox cannot run, or
    when the temporary directory is not safe.
    """
    if policy.mode == "off":
        return None
    if not available():
        if policy.mode == "required":
            raise SandboxUnavailableError("The shell sandbox is required but is not available")
        return None
    if policy.temp_root is None:
        raise SandboxUnavailableError("The shell sandbox has no directory for temporary files")
    temp_dir = session_temp_dir(policy.temp_root, session_id, project_root)
    read_paths = [*policy.runtime_read_paths, *policy.extra_read_paths]
    text = profile(project_root, temp_dir, read_paths, allow_network=policy.allow_network)
    # HOME points to the private directory: tools do not read the real user configuration.
    environment = {
        "PATH": trusted_path(read_paths),
        "HOME": str(temp_dir),
        "TMPDIR": str(temp_dir),
        "TMP": str(temp_dir),
        "TEMP": str(temp_dir),
        "XDG_CONFIG_HOME": str(temp_dir / ".config"),
        "XDG_CACHE_HOME": str(temp_dir / ".cache"),
        # The system git configuration stays closed; git must not fail on it.
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
    }
    return [SANDBOX_EXEC, "-p", text], environment


SYSTEM_BIN = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")


def trusted_path(read_paths: Sequence[str]) -> str:
    """Return a short PATH built from trusted directories only.

    The user PATH is not used: it names directories that the sandbox closes, and a program
    search stops with an access error at the first such directory. The real developer tools
    come first, because /usr/bin/git and similar programs are xcrun launchers that write to
    the shared user temporary directory.
    """
    parts: list[str] = []
    if os.path.islink(XCODE_SELECT_LINK):
        parts.append(os.path.join(os.path.realpath(XCODE_SELECT_LINK), "usr", "bin"))
    parts.extend(os.path.join(path, "bin") for path in read_paths)
    parts.extend(SYSTEM_BIN)
    unique: list[str] = []
    for part in parts:
        if part not in unique and os.path.isdir(part):
            unique.append(part)
    return os.pathsep.join(unique)


def policy_from_settings(settings: Settings) -> SandboxPolicy:
    """Return the shell sandbox policy that the settings select."""
    return SandboxPolicy(
        mode=settings.shell_sandbox,
        temp_root=settings.data_dir / "sandbox",
        extra_read_paths=tuple(settings.shell_sandbox_read_paths),
        allow_network=settings.shell_sandbox_network,
    )

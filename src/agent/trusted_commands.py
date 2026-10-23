"""Read-only commands that may run without approval.

A command is trusted only when it is plain: one program from a short list of
read-only programs, no shell syntax, and no argument that names a secret file.
Every other command keeps the normal approval; any doubt keeps the approval.
The output of a trusted command goes to the model, so secret files are refused
even when the program only reads.
"""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import shutil
from pathlib import Path, PurePosixPath

from agent.security import is_sensitive_path

# Programs that only read and print. Programs that can run other programs (env,
# command, awk, find, xargs, man), write files (sort -o, uniq, file -C), change the
# system (date, hostname) or show the command lines of other users (ps) are not here.
TRUSTED_PROGRAMS = frozenset(
    {
        "basename",
        "cat",
        "df",
        "dirname",
        "du",
        "free",
        "grep",
        "head",
        "id",
        "ls",
        "md5sum",
        "pwd",
        "readlink",
        "realpath",
        "sha1sum",
        "sha256sum",
        "stat",
        "tail",
        "uname",
        "uptime",
        "wc",
        "which",
        "whoami",
    }
)

_CONTENT_READERS = frozenset(
    {"cat", "grep", "head", "md5sum", "sha1sum", "sha256sum", "tail", "wc"}
)

# Characters with a meaning for a shell: chaining, pipes, redirection, substitution,
# globs, home and history expansion, comments, escapes and new lines.
_SHELL_SYNTAX = frozenset(";&|<>$`(){}[]*?!~#\\\n\r")

# Path parts and names that hold secrets, in addition to is_sensitive_path.
_SECRET_PARTS = frozenset(
    {
        ".aws",
        ".azure",
        ".docker",
        ".git-credentials",
        ".gnupg",
        ".kube",
        ".pgpass",
        ".ssh",
        "gcloud",
        "gshadow",
        "secrets",
        "shadow",
        "sudoers",
        "sudoers.d",
    }
)
_SECRET_SUFFIXES = frozenset({".kdbx", ".pem", ".tfstate", ".tfvars"})
# Names and neighbouring path parts that keep tokens or passwords in plain text: an
# environment file of direnv, a state backup of Terraform, the hosts file of the
# GitHub CLI and a git configuration (remote URLs can carry credentials).
_SECRET_NAMES = frozenset({".envrc"})
_SECRET_NAME_ENDINGS = (".tfstate.backup",)
_SECRET_PAIRS = frozenset({(".git", "config"), (".config", "gh")})

_SHORT_FLAGS = {
    "basename": "az",
    "cat": "AbEenstTuv",
    "df": "ahiHklPT",
    "dirname": "z",
    "du": "achklsSx",
    "free": "bghkmw",
    "grep": "bcEFiIlLnNoqsvVwxZ",
    "head": "qv",
    "id": "Gnru",
    "ls": "aAbBcCdDfFgGhHiIklnNopqQrRsStTuUvx1",
    "md5sum": "bt",
    "pwd": "LP",
    "readlink": "efmnqsvz",
    "realpath": "emqsz",
    "sha1sum": "bt",
    "sha256sum": "bt",
    "stat": "t",
    "tail": "qv",
    "uname": "aimnoprsv",
    "uptime": "ps",
    "wc": "cLlmw",
    "which": "a",
}
_LONG_FLAGS = {
    "basename": {"--multiple", "--zero"},
    "cat": {"--number", "--number-nonblank"},
    "df": {"--human-readable", "--inodes", "--local", "--portability"},
    "dirname": {"--zero"},
    "du": {"--human-readable", "--summarize", "--total"},
    "free": {"--bytes", "--gibi", "--human", "--kibi", "--mebi", "--wide"},
    "grep": {"--count", "--fixed-strings", "--ignore-case", "--line-number", "--word-regexp"},
    "head": {"--quiet", "--verbose"},
    "id": {"--groups", "--name", "--real", "--user"},
    "ls": {"--all", "--almost-all", "--directory", "--human-readable", "--numeric-uid-gid"},
    "md5sum": {"--binary", "--text"},
    "pwd": {"--logical", "--physical"},
    "readlink": {"--canonicalize", "--canonicalize-existing", "--canonicalize-missing"},
    "realpath": {"--canonicalize-existing", "--canonicalize-missing", "--quiet"},
    "sha1sum": {"--binary", "--text"},
    "sha256sum": {"--binary", "--text"},
    "stat": {"--terse"},
    "tail": {"--quiet", "--verbose"},
    "uname": {"--all"},
    "uptime": {"--pretty", "--since"},
    "wc": {"--bytes", "--chars", "--lines", "--words"},
}


def _safe_arguments(words: list[str]) -> bool:
    """Allow only understood options; indirect file lists and recursion need approval."""
    program = words[0]
    operands_only = False
    numeric_value = False
    for word in words[1:]:
        if numeric_value:
            if re.fullmatch(r"[0-9]+", word) is None:
                return False
            numeric_value = False
        elif operands_only or not word.startswith("-") or word == "-":
            if _secret(word):
                return False
        elif word == "--":
            operands_only = True
        elif program in {"head", "tail"} and word in {"-n", "-c", "--lines", "--bytes"}:
            numeric_value = True
        elif program in {"head", "tail"} and re.fullmatch(
            r"(?:-[nc]?|--(?:lines|bytes)=)[0-9]+", word
        ):
            continue
        elif word.startswith("--"):
            if word not in _LONG_FLAGS.get(program, set()):
                return False
        elif not word[1:] or any(flag not in _SHORT_FLAGS.get(program, "") for flag in word[1:]):
            return False
    return not numeric_value


def trusted_remote_command(command: str) -> bool:
    """Trust remote metadata only; remote content paths cannot be checked for secret links."""
    words = _plain_words(command)
    return words is not None and words[0] not in _CONTENT_READERS


def trusted_program_path(program: str) -> str | None:
    """Resolve trusted programs independently of the inherited executable search path."""
    if program not in TRUSTED_PROGRAMS or os.name != "posix":
        return None
    return shutil.which(program, path=os.defpath)


def trusted_local_command(command: str, project_root: Path) -> bool:
    """Return True when a local command may run without approval.

    Every path in the command must stay inside the project, as for the file tools:
    no absolute path, no parent traversal and no symbolic link that leads outside.
    """
    words = _plain_words(command)
    if words is None or trusted_program_path(words[0]) is None:
        return False
    root = project_root.resolve()
    for word in words[1:]:
        value = _value(word)
        if not value:
            continue
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            return False
        candidate = root / path
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        if resolved != root and root not in resolved.parents:
            return False
        if _secret(str(resolved.relative_to(root))):
            return False
    return True


def _plain_words(command: str) -> list[str] | None:
    """Return the words of a plain trusted command, or None."""
    if not command.strip() or any(char in _SHELL_SYNTAX for char in command):
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or words[0] not in TRUSTED_PROGRAMS:
        return None
    if not _safe_arguments(words):
        return None
    return words


def _value(word: str) -> str:
    """Return the value of an option such as --file=VALUE, or the word itself."""
    if word.startswith("-") and "=" in word:
        return word.split("=", 1)[1]
    return word


def _secret(value: str) -> bool:
    if not value:
        return False
    path = PurePosixPath(value)
    normalized = PurePosixPath(posixpath.normpath(value))
    if is_sensitive_path(Path(value)):
        return True
    if normalized.parts[:2] in {("/", "proc"), ("/", "dev"), ("/", "sys")}:
        return True
    parts = [part.lower() for part in path.parts]
    for lowered in parts:
        if lowered in _SECRET_PARTS or lowered in _SECRET_NAMES:
            return True
        if lowered.endswith(("_key", "_history", *_SECRET_NAME_ENDINGS)):
            return True
        if lowered.startswith("ssh_host_"):
            return True
    if any(pair in _SECRET_PAIRS for pair in zip(parts, parts[1:], strict=False)):
        return True
    return path.suffix.lower() in _SECRET_SUFFIXES

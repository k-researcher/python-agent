# ADR 0006: Portable file operations layer for Windows

Russian version: [0006-windows-file-fallback.md](0006-windows-file-fallback.md).

## Status

Proposed. The code is written, but the acceptance run on Windows is not done yet. Windows
support is declared after it. Nothing changes on macOS and Linux.

## Context

`SafeProjectFS` in `src/agent/safe_fs.py` works through directory descriptors: `dir_fd`,
`O_NOFOLLOW`, a hard link for exclusive publication, directory fsync. The constructor requires
these primitives from the runtime. CPython on Windows does not provide them, and the constructor
refuses to run. This is a limit of the runtime, not of the file system: NTFS supports hard links.

The tools that depend on the layer are `read_many`, `write_file`, `edit_file`, `multi_edit`,
`remove_file`, `apply_patch` and `edit_by_lines`, and the creation and update of skills. On
Windows all of them fail with one error. In `run_checks` the error is swallowed while the
configuration is read, and the user sees "No project check commands were detected for the
requested kind".

The layer error named internal agent primitives. An observation from use (November 2026, one
user): the model did not connect the error with the platform and told the user to lift a
non-existent "Windows security restriction" and to create the file by hand.

An implementation on Windows handles (`ReplaceFileW`, DACL) is rejected: a third layer with a
new error surface under the same contract requirements. The portable layer reuses the contract
and the shared checks. The native option is revisited if acceptance reveals unacceptable limits
of the portable layer.

## Decision

1. Code: `src/agent/safe_fs.py`. The base class `ProjectFS`, the layer `PortableProjectFS` and
   the factory `project_fs(root)`.
2. The factory selects the layer by the capabilities of the runtime. Full support of the POSIX
   primitives: the existing `SafeProjectFS`. Windows: `PortableProjectFS`, when the root is on
   a local drive with NTFS. Other runtimes, POSIX with incomplete support and a root outside
   local NTFS give an explicit refusal that names the reason. Protection on POSIX is not
   weakened automatically.
3. `ProjectFS` declares the nine operations of the layer and the shared path validation:
   project root, no `..`, no absolute paths outside the root, no secret-bearing paths. The types
   and texts of `MissingFileError`, `PathSecurityError` and `ToolError` stay: patch rollback
   tells them apart by type, the skill library by content.
4. The portable layer examines the raw argument before any path resolution and rejects: a
   drive-relative path (`C:name`), a path from the drive root (`\name`), UNC paths and device
   namespaces (`\\server`, `\\.\`, `\\?\`, named pipes included); reserved device names (`CON`,
   `CONIN$`, `CONOUT$`, `NUL`, `PRN`, `AUX`, `COM1`–`COM9`, `LPT1`–`LPT9` and the superscript
   spellings, case-insensitive, also with an extension and with a space before it); a colon
   inside a component (ADS streams); trailing dots and spaces; the tail of an 8.3 short name;
   a symlink, a junction and other reparse points, by the entry attribute
   `FILE_ATTRIBUTE_REPARSE_POINT`.
5. Every existing component below the root must match a real directory entry: exactly, or by one
   case variant. Several variants mean a case-sensitive directory, none mean an alias of the
   name; both are rejected. The full path is bounded so that the temporary file stays inside
   `MAX_PATH`.
6. The root is resolved once from the trusted operator configuration. A junction in the root
   itself and its ancestors is the operator's trust zone, like a bind mount on POSIX.
7. A write goes to a temporary `os.open` file with `O_CREAT | O_EXCL | O_BINARY` in the same
   directory: write, `fsync`, close the descriptor, check the target again, then `os.replace`
   (inside — `MoveFileExW` with replacement). A read is opened in binary mode too.
8. The file version is `(st_dev, st_ino, st_size, mtime_ns)`. It is compared at three points:
   the directory entry against the open descriptor, after the read, and immediately before a
   conditional replacement or removal. When the file identifier is unavailable, conditional
   operations refuse; a plain write stays available.
9. Exclusive publication: a fully written and closed temporary file through `os.rename`. On
   Windows it refuses a taken name, so a finished file is published without overwriting. On
   POSIX `os.rename` replaces the target, so the rule applies in the portable layer only.
10. Modes and metadata are not carried over. On POSIX `fchmod` is mandatory: its failure aborts
    the replacement. On Windows file modes are not preserved and directory fsync does not run.
11. The layer is created through the factory in `src/agent/tools/builtin.py`,
    `src/agent/tools/patch_tools.py`, `src/agent/tools/check_tools.py` and
    `src/agent/skill_library.py`. In `read_many` the file layer validates the path first;
    `PathGuard` after it only normalizes the displayed name. `run_checks` creates the layer
    outside the handler of configuration errors.
12. Skills: `load_skill` and `scan_skills` refuse a link or reparse point in place of a skill
    directory before the manifest is read; a skill update checks the manifest path through the
    layer before the read. The manifest text for the revision and the comparison is read
    without CRLF normalization.
13. On Windows the `apply_patch` parser keeps trailing dots and spaces in the file name of a
    header, so that the file layer rejects them. On other systems the name is stripped of them,
    as before.
14. The guarantee in the security documents is described separately for POSIX and Windows:
    `docs/security.md`, `docs/threat-model.md` and their English versions.

Not part of the decision, still POSIX-dependent: the import of a skill ZIP archive (it writes
files with `O_NOFOLLOW`), the removal of a skill directory (refused through `shutil.rmtree`),
reading a skill file with the `read_skill` tool, the shell sandbox (ADR 0005) and the external
engines.

## Consequences

- After acceptance, the tools of the layer and the creation and update of skills work on
  Windows for a project on a local NTFS drive, without system configuration.
- macOS and Linux work as before: the factory selects the existing layer.
- A refusal of the layer names the reason: a platform without a layer, or a volume outside
  local NTFS.
- On Windows a replaced file loses its modes, attributes and ADS streams.

## Acceptance

- `tests/unit/test_portable_fs.py` checks the layer on any platform: Windows name forms, links,
  conflicts, a failed exclusive publication, the `apply_patch` rollback, byte-exact UTF-8, the
  length limit and the root check on an imitation of Win32.
- The acceptance run on Windows with local NTFS is not done yet. It includes: a session with
  `write_file`, `edit_file` and `apply_patch`; a real junction inside the project is rejected;
  a collision at `rename` gives "File already exists"; on a replacement refusal the target does
  not change; a skill manifest with CRLF passes the read-update cycle through the API.

## Residual risks

- On Windows the check of a component and the action are separated in time: between `lstat`
  and the operation a component can be replaced with a link or a junction. There is no `dir_fd`
  anchor, and the race remains.
- The modification time on NTFS updates when the writing descriptor closes. A change to a file
  that another process keeps open can stay unseen by the version check.
- The temporary file inherits the permissions of its parent directory at creation, before the
  publication. When they are wider than those of the original file, the content is briefly
  available more widely. The replacement resets an individual file DACL to the inherited one.
- The check of the real name and the ban of reparse points are verified by imitation only: the
  acceptance run confirms 8.3 short names, junctions and case-sensitive NTFS directories.
- In a case-sensitive NTFS directory a request with one case variant opens the existing entry
  in another case.
- `os.replace` refuses when the target is open with a restrictive sharing mode. The refusal
  gives a layer error; the target does not change.
- Hard links are indistinguishable from regular files on both platforms. The same
  acknowledgement for the shell sandbox is ADR 0005.
- The durability of a replacement after an abrupt power loss is not guaranteed.

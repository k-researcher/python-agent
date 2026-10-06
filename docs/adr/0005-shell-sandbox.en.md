# ADR 0005: Operating system sandbox for shell commands

Russian version: [0005-shell-sandbox.md](0005-shell-sandbox.md).

## Status

Accepted. The default mode is `off`. Agent workers must use the `required` mode.

## Context

The agent writes code and runs it through the shell: tests, scripts, migrations. A list of
permitted commands does not give protection. For example, `pytest` runs tests that the agent
wrote. Such a test can read `~/.ssh`, the `.env` file of a different project or the user
configuration, and return the data in the command output. Requirement: the agent code can do
all operations, but only inside the current project.

## Decision

1. A shell command runs through `sandbox-exec` (macOS) with a deny-by-default profile.
   Code: `src/agent/sandbox.py`.
2. Read and write access: the project directory and the temporary directory of the session.
3. Read-only access: a short list of system paths, the interpreter of the agent process and
   the active developer tools directory. The paths `/usr/local` and `/System/Volumes/*` are closed.
4. The network is closed. Process information of other processes, shared memory and POSIX
   semaphores are closed. `sysctl` is open for listed names only. The command input is closed.
5. Only the agent process and the trusted setting `AGENT_SHELL_SANDBOX_READ_PATHS` select the
   permitted paths. Project files do not change the profile, because the agent code can change
   these files.
6. The temporary directory of the session is outside the project. Before each command, the
   server examines it: a real directory, not a link, owned by the current user, mode `0700`.
   If a check fails, the command does not run.
7. `HOME`, `TMPDIR` and `XDG_*` point to the temporary directory of the session. Git does not
   read the system configuration (`GIT_CONFIG_NOSYSTEM`, `GIT_ATTR_NOSYSTEM`).
8. Modes of the setting `AGENT_SHELL_SANDBOX`:
   - `off`: no sandbox;
   - `auto`: the sandbox runs when the system supports it;
   - `required`: a command does not run without a working sandbox.
9. `available()` tests the sandbox with a trial command. A sandbox cannot start inside a
   sandbox, so the test returns `False` there.

## Consequences

- Tests, linters, type checks, migrations and `git` work inside the project without changes.
- Commands that need the network (package installation, `git push`) do not work in the
  sandbox. They need a separate approved step outside the sandbox or
  `AGENT_SHELL_SANDBOX_NETWORK`.
- A program outside the trusted paths does not start. This also applies to the virtual
  environment of the agent itself.
- A git worktree needs a setting that permits reading the `.git` directory of the main repository.
- The sandbox is available on macOS only. On Linux, a container or a separate operating system
  user does this work; a Linux backend is a separate task.

## Residual risks

- A hard link to an outside file that the user made in the project is readable. Code inside
  the sandbox cannot make such a link.
- The list of top-level names in the root directory `/` is visible: the macOS program loader
  needs it.
- The sandbox does not limit processor time, memory or the number of processes.
- A child process that calls `setsid()` can stay alive after the process group stops.

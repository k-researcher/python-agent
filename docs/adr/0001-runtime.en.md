# ADR 0001: Independent Python runtime

## Status

Accepted.

## Context

The agent must run on one workstation with one command. It must keep session state after a
restart. It must stop each side effect until the user approves it.

## Decision

Use FastAPI, asyncio and SQLite in one local process. Serve HTTP and WebSocket on one port.
Keep the session state in the database. Do not keep it in memory or in JSON files.

## Reasons

- One start command on all operating systems.
- Atomic changes of the session state.
- Recovery after a restart.
- No custom WebSocket implementation.
- One central control point for approvals.

## Consequences

The runtime has a bounded local context, child sessions, capability flags, approvals and an
audit of outbound data. SQLite is the default database. PostgreSQL is also permitted through
an async driver.

The single-process mode stays for local work. ADR 0002 describes horizontal scaling with
PostgreSQL, Redis Streams, Pub/Sub and separate workers.

# ADR 0002: Redis Streams and separate workers

## Status

Accepted. This mode is optional. The embedded runtime stays the default.

## Context

One process cannot serve many parallel sessions. More processes must share the session state
and must not run the same session at the same time.

## Decision

In the distributed configuration:

- PostgreSQL keeps the session state.
- A Redis Stream is the durable queue of signals for sessions that are ready to run.
- Redis Pub/Sub carries the temporary UI events.
- The API does not run the agent loop or the tool calls.
- Workers use one consumer group and a distributed lock for each session ID.

## Reliability

A worker acknowledges a job in the stream after the agent loop returns. If a worker does not
acknowledge a job, another worker can claim it after the visibility timeout. The runtime
saves the assistant and tool-call states before each external action. Thus a worker can
continue the loop after a process failure.

After a failure, a tool call with the status `running` has an unknown result. The runtime
writes an error result for it and does not run it again. An automatic retry can repeat a
payment, an SQL update, an SSH command or a file write. Thus the runtime asks the user for a
new explicit action.

## Consequences

- Redis Pub/Sub is not a log. The `events` table is the source of the event history. The
  WebSocket only tells the UI to read the state again.
- The stop of a running network operation is cooperative. Its timeout sets the upper limit.
- The multi-process mode does not permit SQLite.
- Exactly-once execution of an external side effect is not possible unless the target system
  accepts an idempotency key.

# Stream-json contract fixtures

Synthetic, anonymized transcripts for Claude Code 2.1.311. Sources: the accepted
local findings (verified invocation/control request shapes and rate limit/result
fields), installed CLI `--help` and `auth status --help` (2.1.312; help only, NOT
compatibility evidence), and the official Python Agent SDK protocol types/query
implementation (reference only; no SDK dependency). No native auth command or
model turn was run, and no credentials or account packets were captured.
On 2026-10-30, the permitted version/help commands reported installed 2.1.314.
This is also outside the supported range; it does not change the fixture contract.

https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py
https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/_internal/query.py
https://code.claude.com/docs/en/sandboxing
https://code.claude.com/docs/en/settings-reference

`protocol.json` describes known shapes, not a generated or authoritative schema.
`events.json` exercises optional streaming defensively; the invocation deliberately
does not enable unverified partial streaming. Cache creation and cache read tokens
are summed in the neutral port's single cache field. Cost is notional on a plan.

IMPORTANT: Neither findings nor initialize's ordinary response attests effective
sandbox, file-tool restrictions, hooks/plugins or native session identity before
user dispatch. The production gate fails closed. Transport tests explicitly
inject trusted readiness and replace the isolation gate to exercise mapping and
cleanup; separate health tests execute version/auth/initialize over the guard.
The peer's pre-user system init timing is synthetic; these tests do not prove native startup
ordering or availability. No invented effective-settings field grants readiness.
Both 2.1.312 and 2.1.314 are rejected before auth or print mode.

The peer uses real pipes, a private temp and the real guard. PATH is replaced and
the parent permits only this fake executable; network calls are refused in both
processes. Scenario receives act as deterministic barriers, with no sleeps.
The peer emits a test-only startup line. The fixture consumes it before returning
the guarded pipes to the adapter. The guard's own group handshake precedes child
interpreter startup and cannot establish protocol readiness. No timeout is changed.

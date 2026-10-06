# Security and outbound data model

Russian version: [security.md](security.md).

## Outbound data channels

1. The LLM endpoint gets the active context. The local ContextManager selects this context.
2. HTTP and web search operate only after approval and only for hosts in the allowlist.
3. Named database and SSH connections operate only after approval. Secrets stay in the
   configuration.
4. The embedding endpoint exists only if the knowledge base and the embedding URL are
   explicitly enabled.
5. The provider connection test (`/api/settings/models/providers/{id}/test`) sends
   `GET /models`. This request does not contain dialog data.

The server records each attempt in `outbound_audit`. The log keeps metadata and a hash. The
log does not keep the content. Thus you can prove the fact and the size of a transfer without
a second copy of confidential data.

Before the server writes a provider error to the log, to `Session.error` or to events, it
removes the API key and typical key formats from the text.

In the distributed mode, the Redis Stream contains only session IDs. Pub/Sub carries UI events
with service IDs and statuses. Dialog texts stay in PostgreSQL. A worker loads them by session
ID. Do not expose Redis or PostgreSQL to an external network. Compose does not publish their
ports on the host.

## Access to the UI and the API

- The server examines the exact `Host` header. This prevents DNS rebinding.
- A browser logs in with a one-time link. After the login, an `HttpOnly; SameSite=Strict`
  cookie keeps the session. On HTTPS, the cookie has the `__Host-` prefix and the `Secure`
  flag.
- Mutating requests require an allowed `Origin` and the CSRF token.
- The WebSocket examines `Origin` and the cookie before it accepts the connection and again
  every 5 seconds.
- `X-Agent-Token` is only for the CLI, scripts and health checks.

Details: [ADR 0003](adr/0003-browser-authentication.en.md).

## Local boundaries

`SafeProjectFS` writes, edits and removes files through file descriptors. It does not follow a
symlink in any path component. It makes the temporary file in the same directory. The file
replacement is atomic. `PathGuard` prevents `..`, absolute paths outside the project and
symlink escapes for reads.

The shell gets the project root as `cwd`, a limited timeout and a limited output. The shell
always requires approval. The command runs in its own process group. Stop, timeout and output
overflow end the full group. Run the backend as a separate unprivileged user as a second
boundary.

File tools block `.env`, `.npmrc`, private keys and typical credential files. The shell does
not inherit `AGENT_*`, cloud credentials or the SSH agent environment. The memory limit applies
to stdout and stderr. An approved shell can still read the files of the system user and open
network connections. Thus a non-root container user, a read-only root file system and a
careful approval check are mandatory.

## Model keys

`config/models.yaml` does not contain keys. A provider refers to an environment variable
through `api_key_env`. The server encrypts a key from the UI with Fernet. The master key is
`AGENT_SECRET_KEY`. The API never returns a key. It returns only `api_key_set` and the last
four characters. Details: [ADR 0004](adr/0004-model-configuration-layers.en.md).

## Limits of the guarantees

An approval shows the intention of the user. It does not prove that a command or an SQL
statement is safe. Before the approval, the UI shows the tool name, the risk level and the
arguments. For access from an external network, use a TLS reverse proxy. The login link does
not replace TLS, a network firewall or secret rotation.

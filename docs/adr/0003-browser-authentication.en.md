# ADR 0003: Browser authentication with one-time login links

## Status

Accepted. Replaces the manual API token field in the UI.

## Context

The previous UI asked the user to type the API token into a form. The token stayed in the
JavaScript memory and was lost after each page reload. Without a token, the local API
accepted all requests. A malicious web page could use DNS rebinding to send requests to the
local API and approve shell commands. Remote access is a requirement.

## Decision

1. The server examines the exact `Host` header of each request. The allowed names are in
   `AGENT_ALLOWED_HOSTS`. Wildcards are not permitted.
2. A browser logs in with a one-time link. At startup, if no session is active, the server
   writes the link to the console. On localhost, the server also opens the link in the
   browser. The command `agent auth link` makes a new link.
3. The login code is in the URL fragment (`#login=...`). Thus proxies and access logs do not
   get it. The UI removes the fragment and sends the code in a POST request.
4. The server makes a session and sets an `HttpOnly; SameSite=Strict; Path=/` cookie. On
   HTTPS, the cookie name has the `__Host-` prefix and the `Secure` flag.
5. The database keeps only SHA-256 hashes of login codes and session IDs. Login codes are
   single-use. Sessions have an absolute expiry time and can be revoked.
6. Each mutating request must have an allowed `Origin` header and the CSRF token of the
   session in `X-Agent-CSRF`.
7. The WebSocket handshake must have an allowed `Origin` and a valid session cookie. The
   server examines them before it accepts the connection. It examines the session again
   every 5 seconds.
8. `X-Agent-Token` stays for the CLI, scripts and health checks. A valid token does not
   permit a foreign `Origin`.
9. For remote access, `AGENT_PUBLIC_ORIGIN` must use `https://`. Only the reverse proxies in
   `AGENT_TRUSTED_PROXY_IPS` can set `X-Forwarded-*` headers.

## Consequences

- The UI does not show or keep a secret. JavaScript cannot read the session cookie.
- All API instances share the login state through the database. This is necessary for the
  distributed mode.
- After a change of the public origin, users must log in again.
- `/api/health` is public and gives only the service status. Infrastructure details are in
  `/api/ready`, which requires authentication.

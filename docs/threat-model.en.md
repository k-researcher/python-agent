# Threat model

Russian version: [threat-model.md](threat-model.md).

## Assets

- Dialogs, prompts and files of the registered projects.
- Credentials for LLM, SSH, databases and embeddings.
- The permission to do local and remote side effects.
- The integrity of PostgreSQL, the Redis Stream and the knowledge base.

## Trust boundaries

```text
Browser ──cookie + CSRF──> API ──session ID──> Redis Stream ──> Workers
                              └──────────────> PostgreSQL <───────┘
Workers ──approved egress──> LLM / HTTP / DB / SSH / embeddings
Workers ──SafeProjectFS────> /workspace projects
```

LLM output and project content are untrusted data. Redis and PostgreSQL are in the internal
Compose network. The trusted control inputs are the administrator configuration
(`models.yaml`, connections) and the decision of the user at approval.

## Main threats and controls

| Threat | Controls |
| --- | --- |
| Prompt injection causes a side effect | Central risk registry. Approval before execution. JSON Schema validation of arguments before approval |
| Key theft through `read_file` or the shell | Block of files with secrets. Shell without inherited secrets. Non-root container |
| Escape from the project through `..` or a symlink | `SafeProjectFS`: descriptors, `O_NOFOLLOW`, atomic replacement. `PathGuard` for reads |
| SSRF through the HTTP tool | Host allowlist. DNS and private IP range checks. Redirects off. `Host` header not permitted |
| DNS rebinding to the local API | Exact `Host` check. `Origin` check. Cookie session and CSRF token |
| Theft or forgery of a browser login | One-time link in the URL fragment. Hashes of codes and sessions in the database. `HttpOnly; SameSite=Strict` cookie; on HTTPS `__Host-` and `Secure`. Session revocation |
| Side effect after Stop | Stop check before each tool. Atomic claim of the call. Cancellation of pending approvals. End of the shell process group. Stop watcher in the worker |
| Repeated side effect after a worker failure | A call with the status `running` does not run again. The result is an error with an unknown outcome |
| Two workers run one session | Redis lock with TTL and heartbeat. Queue deduplication for each session |
| Lost job after a failure | Redis Stream consumer group. ACK after the loop. `XAUTOCLAIM` for stale jobs |
| Prompt leak through the queue | The stream contains only session IDs. Texts stay in PostgreSQL |
| Key leak through a provider error | Removal of the key and typical key formats from the error text before the log, the session and events |
| DoS through output, regex or XLSX | Limits. Process termination. Regex timeout. Zip size and part limits |
| Replacement of a model or key | `models.yaml` and the protected settings API define models. UI keys are encrypted. A session keeps only the model ID |

## Residual risks

- An approved shell, SSH or SQL operation is powerful. An approval does not prove that it is
  safe.
- The server examines DNS before a request. The DNS of allowlisted hosts stays part of the
  trusted base.
- The login is for one user. There are no roles and no tenant isolation.
- Pub/Sub events are temporary. The UI restores the state from PostgreSQL.
- Exactly-once execution in an external system is not possible without its idempotency key.
- The deployment perimeter supplies TLS, rate limits, backups, log retention and a secret
  manager. The built-in FastAPI runtime does not.

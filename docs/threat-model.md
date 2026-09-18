# Threat model

## Активы

- диалоги, prompts и файлы зарегистрированных проектов;
- LLM, SSH, database и embedding credentials;
- право выполнять локальные и удалённые побочные действия;
- целостность PostgreSQL, Redis Stream и knowledge base.

## Границы доверия

```text
Browser ──token──> API ──session ID──> Redis Stream ──> Workers
                      └──────────────> PostgreSQL <───────┘
Workers ──approved egress──> LLM / HTTP / DB / SSH / embeddings
Workers ──PathGuard────────> /workspace projects
```

LLM output и содержимое проекта считаются недоверенными. Redis/PostgreSQL находятся
во внутренней Compose-сети. Администраторская конфигурация profiles/connections и
решение пользователя в Approval являются доверенными управляющими входами.

## Основные угрозы и меры

| Угроза | Меры |
| --- | --- |
| Prompt injection вызывает side effect | Централизованный risk registry и Approval до выполнения |
| Кража ключей через `read_file`/shell | Блокировка secret-bearing paths; shell без inherited secrets; non-root container |
| Path traversal/symlink escape | Canonical resolve + проверка принадлежности project/frontend root |
| SSRF через HTTP tool | Явный hostname allowlist, DNS/IP private-range checks, redirects off, запрещённый Host header |
| Повтор side effect после падения worker | `running` tool call не retry; возвращается uncertain error |
| Два worker ведут одну сессию | Redis token lock с TTL heartbeat и session-level queue dedupe |
| Потеря job при падении | Redis Stream consumer group, ACK после цикла, XAUTOCLAIM stale jobs |
| Утечка prompts через очередь | Stream содержит только session ID; тексты находятся в PostgreSQL |
| DoS через output/regex/XLSX | Limits, process kill, regex timeout/thread, zip size/parts limits |
| Перехват внешнего API | Localhost bind по умолчанию, API token, рекомендация TLS reverse proxy |
| Подмена модели/ключа | Профили задаёт администратор; сессия хранит только проверенное имя профиля |

## Остаточные риски

- Одобренный shell/SSH/SQL является мощной операцией; Approval не доказывает её
  безопасность.
- DNS проверяется перед запросом, но доверенный allowlisted DNS всё равно остаётся
  частью trusted computing base.
- API token — single-user authentication без ролей и tenant isolation.
- Pub/Sub события эфемерны; UI восстанавливает истину из PostgreSQL events/state.
- Exactly-once для произвольной внешней системы невозможно без её idempotency key.
- TLS, rate limiting, backups, log retention и secret manager обеспечивает deployment
  perimeter, а не встроенный FastAPI runtime.

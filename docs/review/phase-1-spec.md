# Фаза 1 — техническая спецификация

## A. Аутентификация и reverse proxy

**Решение:** серверная cookie-сессия обязательна для браузера, включая localhost. Это намеренно строже предложения `docs/review/2026-10-06-architecture-review.md:218`: отсутствие токена в UI не означает анонимный API. Система остаётся однопользовательской, без ролей.

**Точки изменения:** `src/agent/main.py:35`, `src/agent/api.py:364`, `src/agent/config.py:77`. Новые файлы: `src/agent/auth.py`, `src/agent/auth_api.py`, `src/agent/auth_middleware.py`, `src/agent/cli.py`.

### Конфигурация и проверки

Добавить в `Settings`:

```python
allowed_hosts: list[str] = ["localhost", "127.0.0.1", "[::1]"]
public_origin: str | None = None
trusted_proxy_ips: list[str] = []
auth_link_ttl_seconds: int = 600
auth_session_ttl_seconds: int = 604800
```

- `AGENT_ALLOWED_HOSTS` — JSON-массив точных hostnames без схемы/порта; запретить `*` и wildcard-домены. Подключить `TrustedHostMiddleware(..., www_redirect=False)` снаружи авторизации; чужой HTTP Host → `400`. 
- `AGENT_PUBLIC_ORIGIN` — единственный разрешённый browser origin: `scheme://host[:port]`, без credentials, path, query и fragment. Для локального режима default — `http://127.0.0.1:<port>`; для удалённого доступа обязателен HTTPS-origin. Его hostname должен входить в `allowed_hosts`.
- Origin сравнивать как нормализованную тройку scheme/hostname/effective-port, не через suffix-match. Проверять REST `POST/PUT/PATCH/DELETE`, включая auth, и WS **до `accept()`**. Чужой, `null`, некорректный или отсутствующий browser Origin → `403`.
- Исключение отсутствующего Origin — только успешно проверенный `X-Agent-Token`. Присутствующий чужой Origin отвергать даже с правильным токеном. CORS между UI и API не требуется: публикация same-origin.

`https://llm.example.com` (LLM-шлюз) — **LLM destination, не автоматически origin интерфейса**.

Reverse proxy сохраняет исходный `Host`, перезаписывает клиентские `X-Forwarded-For/Proto`, поддерживает WS upgrade. Backend закрыт firewall/приватной сетью. В `src/agent/main.py:82` передавать Uvicorn `proxy_headers=True`, `forwarded_allow_ips` из явного allowlist, не `*`. Uvicorn обрабатывает `For/Proto`; не использовать `X-Forwarded-Host` для обхода Host-проверки или формирования login-link. Ссылка и cookie-политика определяются `public_origin`. 

### БД и bootstrap

Миграция после `0004`:

- `auth_sessions`: `id_hash CHAR(64) PRIMARY KEY`, `created_at`, `expires_at`, nullable `revoked_at`, `user_agent VARCHAR(512)`, `last_seen`, `csrf_token VARCHAR(64)`; индекс `expires_at`.
- `auth_bootstrap_codes`: `code_hash CHAR(64) PRIMARY KEY`, `created_at`, `expires_at`, nullable `consumed_at/revoked_at`.
- `auth_bootstrap_state`: singleton для однократного автоматического выпуска первой ссылки.

Все timestamps — UTC, включая нормализацию SQLite. Session ID и OTP — независимые `secrets.token_urlsafe(32)`; в БД хранить SHA-256. CSRF-токен — отдельное случайное значение, не credential доступа. `user_agent` — метаданные, не жёсткая привязка. Сессия имеет абсолютный TTL; `last_seen` обновлять не чаще раза в минуту. 

Контракты сервиса:

```python
async def issue_link(db: AsyncSession, origin: str, *, ttl_seconds: int) -> BootstrapLink
async def consume_link(db: AsyncSession, code: str, user_agent: str) -> IssuedSession
async def authenticate(db: AsyncSession, raw_id: str) -> AuthSession | None
async def revoke(db: AsyncSession, id_hash: str | None) -> int
```

- Первый API-instance атомарно захватывает singleton и создаёт OTP; после commit печатает в операторскую консоль/лог `PUBLIC_ORIGIN/#code=<otp>`. Остальные instances ничего не выпускают. Workers bootstrap не выполняют.
- `POST /auth/bootstrap {"code":"..."}`: JSON + правильный Origin, без существующей сессии/CSRF. Условный `UPDATE ... WHERE consumed_at IS NULL AND revoked_at IS NULL AND expires_at > now RETURNING ...` и создание сессии — **одна транзакция**. Два конкурентных обмена: один успешный. Неверный/использованный/просроченный код → одинаковый `401`.
- Код не принимать в query. Не логировать тело обмена, cookies или auth headers. Auth-ответы: `Cache-Control: no-store`; bootstrap-страница: `Referrer-Policy: no-referrer`. На proxy ограничить частоту обменов.
- `agent auth link [--ttl-seconds 600]` выпускает новую ссылку непосредственно через БД; не отзывает существующие сессии. CLI не должен импортировать `agent.main`.
- OTP и сессии никогда не размещать в process memory или Redis как единственном хранилище.

### Cookie, CSRF, logout

HTTPS: `__Host-agent_session`; локальный HTTP: `agent_session`. Обязательно `HttpOnly; SameSite=Strict; Path=/`, без `Domain`; `Secure` при HTTPS-origin, `Max-Age` не больше TTL. Принимать только cookie выбранного режима; отвергать неоднозначные дубликаты. `__Host-` требует `Secure`, `Path=/` и отсутствия `Domain`. 

- `GET /auth/session` → `{authenticated, expires_at?, csrf_token?}`; unauthenticated-вариант не раскрывает конфигурацию.
- Cookie-auth мутации требуют JSON и `X-Agent-CSRF`, constant-time сравнение с session CSRF. Bootstrap исключён только из CSRF, не из Origin/JSON. CSRF хранить во frontend только в памяти. SameSite не заменяет эту проверку. 
- `POST /auth/logout` → отзыв текущей сессии, очистка cookie, `204`.
- `POST /auth/revoke {"all":true}` → отзыв всех auth-сессий, включая текущую; доступ только после auth + CSRF.
- Защитить `/api/**`, `/docs`, `/redoc`, `/openapi.json`. Статика, bootstrap и минимальный auth-status доступны без сессии.
- `X-Agent-Token` оставить для CLI/healthcheck; правильный header освобождает от CSRF, но не Host-проверки. Неверный явно переданный токен → `401`, без fallback к cookie.
- WS: удалить первый auth-frame. Проверить Origin и cookie/header до `accept`; отказ до handshake. После подключения проверять expiry/revoke минимум каждые пять секунд, закрывать `4401`. Logout должен закрывать сокеты на других API-instances через проверку БД.

### Frontend

`frontend/src/api.ts:77`: удалить `apiToken`, `setApiToken`, `websocketToken`; `request<T>` использует `credentials:"same-origin"`, добавляет CSRF только к мутациям и сохраняет HTTP status в `ApiError`.

Новый `frontend/src/auth.ts`: `bootstrapFromFragment(): Promise<void>`, `refreshAuth(): Promise<AuthState>`. Считать fragment и **сразу** очистить URL через `history.replaceState`, затем обменять код POST.

`frontend/src/App.vue:75`, `frontend/src/App.vue:137`, `frontend/src/App.vue:183`: удалить ввод токена; сначала auth-check, затем загрузка данных и cookie-WS. При `401` очищать данные/CSRF и показывать «Нужна ссылка входа: выполните `agent auth link`». На WS-disconnect сначала проверять auth, не бесконечно reconnect. Добавить logout. Для Vite проксировать `/auth` вместе с `/api`; локальный public origin — адрес Vite.

## B. B15: безопасные файловые мутации

Уязвимые операции: `src/agent/tools/builtin.py:155`, `src/agent/tools/builtin.py:187`, `src/agent/tools/builtin.py:215`. Одного исправления `exists()` недостаточно: остаётся TOCTOU.

Новый `src/agent/safe_fs.py`:

```python
class SafeProjectFS:
    def __init__(self, root: Path) -> None: ...
    def write_text(self, path: str, content: str) -> int: ...
    def edit_text(self, path: str, old: str, new: str, *, replace_all: bool) -> int: ...
    def remove_file(self, path: str) -> None: ...
```

Алгоритм:

1. Преобразовать допустимый абсолютный путь в относительный **лексически**, не разрешая ссылки. Отвергать NUL, `..`, выход за root и sensitive paths согласно `src/agent/security.py:10`.
2. Открыть root directory FD. Каждый компонент родителя открывать через `os.open(component, O_DIRECTORY|O_NOFOLLOW, dir_fd=parent_fd)`; удерживать FD до завершения. Запретить symlink во всех компонентах, включая внутренние.
3. Финальный компонент проверять `lstat` относительно FD. Для чтения edit — `O_RDONLY|O_NOFOLLOW`, затем `fstat`: только regular file, существующий лимит 1 МБ. Не читать FIFO/device.
4. Write/edit: создать случайный temp в **той же директории**, `O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0600`; записать, `fsync`, сохранить обычные permission bits существующего файла. Повторно проверить target; edit при изменении inode/mtime/size → конфликт.
5. `os.replace(temp, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)`, затем directory `fsync`, где поддерживается. Ошибки удаляют temp и закрывают FD. Rename не следует финальному symlink и атомарен на одной файловой системе. 
6. Remove: `lstat` regular file и `os.unlink(name, dir_fd=parent_fd)`; никаких recursive deletes или `resolve()` перед unlink.

Использовать общий POSIX-путь на macOS/Linux, без зависимости от Linux-only `openat2`. При отсутствии необходимых primitives — отказ, не небезопасный fallback. Создание отсутствующих родителей — через FD-relative `mkdir` с последующим безопасным open.

Граница гарантии: защита от symlink-перенаправления, не OS sandbox против другого процесса того же UID, перемещающего уже открытые каталоги.

## C. B16/B17: Stop и процессы

Точки: `src/agent/runtime.py:79`, `src/agent/runtime.py:116`, `src/agent/runtime.py:245`, `src/agent/runtime.py:409`, `src/agent/worker.py:86`.

Добавить `ApprovalStatus.cancelled`, `ToolCallStatus.cancelled` и `Session.run_generation INTEGER NOT NULL DEFAULT 0` — `src/agent/models.py:30`, `src/agent/models.py:39`, `src/agent/models.py:60`.

**Транзакция Stop:** блокировка Session → ToolCall → Approval; `stop_requested=True`, `status=stopped`, увеличение generation, отмена pending approvals с `resolved_at`, отмена ещё не начатых calls, durable Event. Затем cancel локальных задач и publish. Потомки останавливаются также; создание ребёнка блокирует/проверяет parent в своей транзакции. Повторный Stop идемпотентен.

`resolve_approval` выполняет проверку Session, изменение approval/call и возможный переход в pending **одной транзакцией**. Stopped/stop_requested/cancelled → `409`; никакого enqueue. В SQLite использовать conditional updates/`BEGIN IMMEDIATE`, не полагаться на `FOR UPDATE`.

```python
async def _check_running(self, session_id: str, generation: int) -> None
async def run_job(self, session_id: str, generation: int) -> None
async def _watch_stop(self, session_id: str, generation: int,
                      run_task: asyncio.Task[None]) -> None
```

Проверки — перед каждым tool, перед его execution claim, перед LLM и записью результата. Использовать свежие SELECT/короткие транзакции, не старый ORM-object. Удалить безусловный сброс `stop_requested` на `src/agent/runtime.py:247`; все статусные переходы и commits результатов fence-ить generation.

Явное продолжение через start/message создаёт новое generation и очищает stop; старые cancelled calls не возобновляются. Generation передавать в Redis job и dedupe-key, иначе старый job/ACK затронет новый запуск.

**Redis: гибрид предпочтителен.** БД — источник истины; watchdog активного execution проверяет её каждые 500 мс. Канал `<namespace>:cancel` ускоряет отмену; payload содержит session/generation, получатель перепроверяет БД. При потерянном publish/reconnect работает polling; при невозможности проверить БД запрещать новые tools и отменять execution. Pub/sub alone непригоден: доставка at-most-once. 

`AgentWorker` отслеживает run tasks по session/generation. После подтверждённого Stop и cleanup ACK выполняется; shutdown/неожиданная отмена сохраняет pending job. Lock освобождается только после ожидания cleanup.

`src/agent/tools/builtin.py:259`: `start_new_session=True`; PGID равен PID.

```python
async def terminate_process_group(process: asyncio.subprocess.Process,
                                  *, grace_seconds: float = 2.0) -> None
```

При timeout/`CancelledError`: `killpg(SIGTERM)`, затем `SIGKILL` оставшейся группе; проверять группу даже при завершившемся прямом потомке. Cleanup — отдельная shielded task, которую действительно дождаться; ограниченно дочитать pipes, дождаться process и reader tasks. Исходный `CancelledError` пробросить; timeout → `ToolError`. Иначе ожидание процесса с заполненными pipes может зависнуть. 

Stop подтверждает durable запрос, **не откат уже выполненного эффекта**. Уже допущенная атомарная операция может завершиться; cancellation незавершённого tool сохраняет `side_effects_unknown=true`. Изменение политики output-limit/HOME сюда не включать.

UI Stop должен быть доступен и для `awaiting_confirmation` — `frontend/src/App.vue:254`.

## D. `config/models.yaml`

Новые `src/agent/model_config.py`, `config/models.example.yaml`. Выбрать собственный safe YAML loader: удобнее обеспечить line/field diagnostics и явную legacy-политику, чем смешивать весь `Settings` с YAML sources. `YamlConfigSettingsSource` допустим как транспорт, но не заменяет строгую схему. 

Все Pydantic-модели: `extra="forbid"`, immutable; запрет дубликатов YAML-ключей, произвольных tags и файлов больше 1 МБ. PyYAML объявить прямой зависимостью.

**Схема:**
- `ProviderSpec`: `kind: Literal["openai","anthropic","jev"]`, `base_url: str`, `api_key_env: str|None`, положительный timeout, retries `0..10`.
- `ReasoningSpec`: `supported: bool`, `allowed_efforts: list[str]`, `default_effort: str|None`, явный поддерживаемый `wire_parameter`.
- `ModelSpec`: `provider: str`, `model: str`, `context_window >=4096`, `max_tokens >0`, reasoning, optional pricing с неотрицательными тарифами.
- `RoutingSpec`: `default: str`, `fallback: list[str]`, `roles: dict[str,str]`.
- `DecisionsSpec`: `enabled=False`, optional provider/model.
- `ModelsConfig`: providers/models maps, routing/decisions; IDs моделей — прежние значения `llm_profile`, длина `1..100`.

Валидировать все ссылки; fallback содержит **model IDs**, не provider IDs. Поэтому `local` из `docs/review/2026-10-06-architecture-review.md:310` требует отдельной модели. Проверять допустимый reasoning effort и `max(reserved_tokens,max_tokens) < context_window`.

Фаза 1 реализует OpenAI-compatible transport. Anthropic/Jev допускаются только как неиспользуемые декларации; выбор такого transport и `decisions.enabled=true` — явная ошибка unsupported, не молчаливое игнорирование. Jev execution остаётся фазой 4.5.

```python
def load_models(settings: Settings, env: Mapping[str, str]) -> ResolvedModelRegistry
def resolve_model(registry: ResolvedModelRegistry, model_id: str) -> ResolvedModel
class LLMClient:
    def __init__(self, model: ResolvedModel) -> None: ...
    async def chat(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]], *,
                   reasoning_effort: str | None = None) -> LLMResponse: ...
```

- `AGENT_MODELS_FILE` default — `PROJECT_ROOT/config/models.yaml`. Указанный явно отсутствующий/ошибочный файл → ошибка.
- Если default-файла нет: адаптер сохраняет текущие `AGENT_LLM_*`/`AGENT_LLM_PROFILES` semantics из `src/agent/config.py:103`; профили получают legacy global context window, выводится deprecation-warning.
- При наличии YAML структура YAML приоритетна; legacy structure variables игнорируются с warning. Явный `AGENT_DEFAULT_LLM_PROFILE` может переопределить default только существующим ID.
- `api_key_env`: process environment → выбранный dotenv. Отсутствующий указанный секрет — ошибка; `null` означает отсутствие Authorization. Legacy пустой ключ сохраняет `configured=false`. Секреты не попадут в YAML dump/errors/public API.
- URL без credentials/query/fragment. Нелокальный `http://` → warning, `agent config check --strict` → failure. Для корпоративного шлюза использовать HTTPS; API-prefix проверить у gateway, не угадывать по hostname.
- `agent config check [--file PATH] [--strict]`: exit `0/1/2` = valid/invalid/warnings-as-errors; ошибки `file:line:column + field`, без secret values. Команда не создаёт директории, БД или supervisor.
- Routing: explicit profile → role default → global default. Fallback последовательно только после исчерпания transient retries; каждый кандидат получает собственный ContextManager и outbound audit; никаких fallback на auth/validation errors или circuit breaker в фазе 1.

Убрать общий context из `src/agent/runtime.py:49`; LLM и context одного execution используют один snapshot. Исправить `/context` — `src/agent/api.py:199`; `public_config` — `src/agent/api.py:66`: сохранить legacy поля, добавить model context/reasoning/config-version; `configured` учитывает keyless providers. Секреты и полный URL не публиковать.

**Reload:** `ModelConfigManager.reload()`, SIGHUP и защищённый `POST /api/admin/reload-config`. Validate-then-swap; ошибка сохраняет предыдущий snapshot. Обновлять cache и enum `run_agent` — `src/agent/runtime.py:495`, `src/agent/tools/advanced.py:395`.

Для Redis записывать generation/checksum конфигурации в БД, уведомлять Redis; все API/workers polling перечитывают одинаково смонтированный YAML. Несовпадение checksum/секретов блокирует новые executions. Уже выполняющиеся сохраняют старый snapshot. Auth/infrastructure Settings reload не затрагивает.

## E. PR, тесты и миграция

**Маленькие PR в порядке зависимости:** Host/Origin + app factory → auth-таблицы/backend/CLI → cookie UI → SafeProjectFS → Stop-транзакции/generation → shell cleanup → Redis cancellation → YAML/schema/config-check → LLM/context/reload/compose. Удалённую сборку выпускать после завершения auth UI.

**Новые/изменённые тесты:**
- Новые `tests/unit/test_auth.py`, `tests/integration/test_auth.py`: Host `400`; foreign/null/missing Origin `403`; spoofed forwarded headers; cookie flags; CSRF; OTP expiry/reuse/concurrent exchange между API-instances; revoke/WS; token healthcheck.
- `tests/unit/test_security.py:25` + новый `test_file_mutations.py`: dangling/internal/external symlink, symlink-parent, детерминированная подмена между проверкой/open/rename; внешний файл неизменён; atomic failure оставляет исходник; temp/FD cleanup.
- `tests/unit/test_shell_safety.py:10`: cancellation/timeout убивают parent и grandchild, включая игнорирование TERM; повторный cancel; сохранение secret isolation/output-limit.
- Новые `test_runtime_stop.py`, `tests/integration/test_stop.py`, `test_worker_stop.py`: Stop между двумя tools, approval после Stop `409`, stop/approve race, stale running commit, descendants, missed pub/sub, DB failure, stale-generation job/ACK. Redis acceptance — реальные PostgreSQL/Redis, не только fakeredis.
- `tests/unit/test_config.py:6`, новый `test_model_config.py`, `tests/unit/test_llm.py:4`, `tests/unit/test_context.py:9`: legacy precedence, duplicate/unknown fields/references, missing secrets, keyless provider, reasoning payload, различные окна, transient fallback.
- Новый `tests/integration/test_config_reload.py`: invalid reload сохраняет snapshot; distributed adoption/mismatch; active execution неизменен; secret redaction.
- `tests/integration/test_api.py:116` переводится на authenticated fixtures; `tests/integration/test_api.py:229` — token/cookie handshake до accept.
- Новый browser E2E: HTTPS cookie-flow, fragment очищен до POST, credentials отсутствуют в storage, logout прекращает reconnect, cancelled approval недоступен.

**Риски:** миграции после существующей `0004`; сначала schema, затем код. Generation меняет Redis wire/dedupe — mixed-version workers запрещены, очередь дренировать при rollout. Старые `llm_profile` IDs сохранить aliases; удаление используемых IDs блокировать. YAML монтировать read-only во все API/workers — `compose.yaml:35`. Смена origin требует повторного входа; atomic replace меняет inode и не сохраняет автоматически ACL/xattrs. Фаза 1 не обещает exactly-once, исправление B4/outbox или остановку процессов, самостоятельно покинувших process group.

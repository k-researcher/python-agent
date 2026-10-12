# Фаза 2 — спецификация ядра


Уже есть pooled `AsyncClient` — `src/agent/llm.py:41`, разбор reasoning — `src/agent/llm.py:128`, effective registry с DB overrides — `src/agent/model_registry.py:54`. Их не реализовывать повторно. Reasoning пока не сохраняется в `Message`; generation отсутствует.

## 1. Общие инварианты и интерфейсы

Новые модули: `execution.py`, `outbox.py`, `history.py`, `reasoning.py`, `usage.py`; `runtime.py` остаётся координатором.

```python
RunFence(session_id: str, generation: int, owner_epoch: int)

async def request_run(db: AsyncSession, session_id: str, reason: str) -> RunRequest
async def run_job(self, request: RunRequest) -> RunOutcome
async def check_running(db: AsyncSession, fence: RunFence) -> None
async def watch_stop(fence: RunFence, task: asyncio.Task[RunOutcome]) -> None
```

- БД — источник истины; Redis — доставка и ускорение уведомлений.
- Транзакции короткие: никаких LLM/tool/network ожиданий внутри открытой транзакции.
- Worker-коммиты сообщений, результатов и статусов проверяют одновременно generation, ownership epoch, действительность lease и отсутствие Stop.
- API-переходы используют блокировку Session и ожидаемое состояние/generation.
- Audit/usage старой попытки разрешено завершить после Stop, но нельзя менять состояние нового запуска.
- `max_inference_steps` сохраняется между approvals/child waits, сбрасывается только при новом запуске. Согласовать поле счётчика с исправлением B1.

Гарантия: **at-least-once доставка, идемпотентные DB-переходы, не exactly-once внешние эффекты**.

## 2. Generation, Redis и transactional outbox

Точки: `src/agent/runtime.py:62`, `src/agent/runtime.py:83`, `src/agent/runtime.py:165`, `src/agent/runtime.py:242`, `src/agent/queue.py:42`, `src/agent/worker.py:86`, `src/agent/events.py:110`.

### Generation и ownership

`run_generation` увеличивается при явном новом запуске/message и при первом Stop текущего запуска. Approval и возврат ребёнка продолжают прежнее generation. Повторный start активной сессии и повторный Stop идемпотентны.

Generation недостаточно для reclaim **внутри одного запуска**: добавить монотонный `owner_epoch`, `owner_token`, `lease_until`. Захват/продление DB lease — conditional update; Redis lock остаётся дополнительным ограничителем. Потеря любого обязательного lease отменяет execution; новый владелец получает новый epoch.

Stop сначала атомарно фиксирует stopped/generation, отменяет approvals и не начатые calls, пишет события/outbox; затем отменяет задачи. Убрать сброс `stop_requested=False` из рабочего claim — `src/agent/runtime.py:300`. `_mark_error` также fence-ить.

Watchdog: свежий DB SELECT каждые **500 мс**, timeout запроса **1 с**; ошибка проверки → fail-closed, cancellation и запрет новых tools. Канал `<namespace>:cancel:v2` содержит session/generation и лишь ускоряет DB-проверку: Pub/Sub допускает потерю уведомлений. 

### B21: единая транзакция

```python
async def append_event(db: AsyncSession, event_type: str,
                       payload: dict[str, Any], session_id: str | None) -> Event
async def dispatch_outbox(batch_size: int = 100) -> int
```

`append_event` делает add/flush, **не commit/broadcast**. Состояние + Event + outbox-intent коммитит вызывающий сервис. Убрать внутренний commit из `src/agent/session_service.py:53`.

Dispatcher захватывает строки короткой транзакцией: PostgreSQL `SKIP LOCKED`, SQLite CAS/claim lease. Публикация выполняется после commit; затем отметка доставки. Crash между отправкой и отметкой допускает повтор.

Добавить сериализованный под Session lock `event_seq`; replay/filter по session и sequence. Глобальный `Event.id` сохранить для совместимости, но не использовать как гарантию порядка commit между конкурентными транзакциями.

Archive/delete — тот же transactional путь (`src/agent/api.py:282`). Удаление запрещено при активном lease/child wait; `session.deleted` — durable tombstone с nullable Event FK и исходным ID в payload/outbox, иначе CASCADE уничтожит уведомление.

### B4: пробуждение не зависит от ACK

Ввести `wake_seq` и `served_seq`; `needs_rerun := wake_seq > served_seq`.

Каждое новое пробуждение атомарно увеличивает `wake_seq` и создаёт enqueue-intent. Job:

```text
protocol_version=2, job_id, session_id, generation, wake_seq, config_checksum
```

Dedupe-ключ — **session/generation/wake_seq**, не только session. Lua объединяет dedupe и `XADD`. Approval-before-ACK получает отдельный ключ, поэтому старый ACK не может поглотить новое пробуждение.

При выходе worker сравнивает обработанный sequence с актуальным; если появилась новая runnable работа — сохраняет повторный enqueue-intent. Это же закрывает embedded-гонку `_tasks` ещё не завершилась → `start=False`.

ACK/XDEL относятся только к конкретному delivery; удаление dedupe проверяет его job ID. Worker с утраченным ownership не ACK-ает reclaimed job того же generation. Устаревшее generation можно утилизировать, не затрагивая новое. Shutdown сохраняет pending delivery; подтверждённый Stop ACK-ается **после cleanup**.

### B19 и восстановление

`claim_stale(..., cursor) -> ClaimPage(next_cursor, messages, deleted_ids)`. Продолжать с возвращённого cursor даже при пустой странице; `0-0` завершает sweep. Чередовать reclaim и новые jobs; ограничить число страниц за iteration. Учитывать Redis 7 deleted IDs; не обрезать непроцессированные задания текущим `MAXLEN`. 

Reconciler восстанавливает runnable сессии с незакрытым wake sequence после Redis потерь/рестартов. Неизвестный результат mutating tool после crash — error/unknown, **без автоматического повторения эффекта**.

**Mixed-version:** maintenance ingress → остановка/drain старых API/workers → schema → отдельные stream/group/namespace `v2` → новые процессы → восстановление durable work. Отсутствующие protocol/generation нельзя трактовать как `0`. Heartbeat публикует protocol/checksum; `/ready` отвергает несовместимый состав. Один version-check нового worker не защищает от старого worker, который его не знает.

## 3. SSE → эфемерные WS-дельты

Точки: `src/agent/llm.py:55`, `src/agent/llm.py:81`, `src/agent/runtime.py:543`, `src/agent/events.py:88`, `src/agent/api.py:387`.

```python
async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]
async def publish_ephemeral(self, envelope: StreamEnvelope) -> None
```

`LLMStreamEvent` — content/reasoning/tool delta либо единственный final с `LLMResponse`. Старый `chat()` собирает этот же поток, отдельного parser не имеет.

Запрос: `stream=true`, поддерживаемый `stream_options.include_usage=true`. Usage может приходить отдельно от содержимого; не считать отсутствие usage нулевой стоимостью. 

Parser поддерживает CRLF, комментарии, многострочные `data:`, `[DONE]`, null-поля, пустые `choices`, error после HTTP 200. Для выбранного choice tool_calls собираются по `index`: аргументы конкатенируются, ID/name сохраняются из соответствующих fragments. JSON/Schema проверяются только после завершения; инструменты никогда не запускаются по дельтам.

Envelope:

```text
type=message.delta|reasoning.delta
session_id, generation, owner_epoch, attempt_id, stream_id, seq, delta
ephemeral=true
```

- `seq` — порядок внутри stream, не durable event cursor.
- Дельты объединять каждые 50 мс/до 8 KiB, передавать напрямую broker; **никаких Event/outbox строк на дельту**.
- Embedded рассылает локально; Redis worker публикует, API слушает Pub/Sub. Потерянные дельты не replay-ятся.
- Успешный final атомарно сохраняет `Message`, reasoning, tool_calls, usage, status и durable событие с `attempt_id/message_id`.
- WS disconnect не отменяет LLM. Reconnect перечитывает durable состояние.
- Медленного subscriber отключать с resync-required, а не молча удалять durable события (`src/agent/events.py:99`). Подписки фильтровать по session; auth перепроверять по таймеру даже при непрерывных дельтах.

**Обрыв:** до terminal finish — interrupted; частичные tools не исполняются. Увиденный partial сохранять однократно как interrupted Message, исключённый из модельного контекста. Retry/fallback получает новый stream ID и `stream.reset`; нельзя склеивать попытки. Terminal finish с последующим обрывом только usage-хвоста допустим как завершённый ответ с неизвестным usage. Crash без final commit оставляет незавершённый attempt для reconciler.

**Audit:** сериализовать окончательный request один раз и отправлять именно эти bytes через `content=...`; хешировать их, включая model/reasoning/stream/max_tokens/tools. На **каждую HTTP-попытку**, включая retries/fallback/summary, записывать started → completed/error/cancelled. Отсутствие started audit блокирует отправку. Секретные headers и полный payload в журнал не помещать. Текущий audit выше внутренних retries этого не обеспечивает.

## 4. Reasoning и wire-адаптеры

Точки: `src/agent/models.py:93`, `src/agent/schemas.py:50`, `src/agent/runtime.py:328`, `src/agent/model_config.py:92`.

Добавить `Message.reasoning_content: Text | None`; возвращать отдельно в `MessageRead`, не смешивать с ответом. Provider-specific signed thinking blocks хранить отдельно в JSON metadata.

**DeepSeek:** V3.2 поддерживает thinking с tools; при продолжении необходимо сохранять исходное reasoning соответствующих assistant-сообщений. Актуальная документация требует передавать reasoning **всех сохранённых ходов при наличии `tools`**; без tools reasoning может игнорироваться. Поэтому нельзя универсально удалять его после следующего user-сообщения. Политика default для DeepSeek — `all_retained_with_tools`, без синтетического reasoning. 

**Подводный камень LiteLLM:** проверенный текущий DeepSeek transformer превращает effort в enabled/disabled, теряя интенсивность; также способен подставлять пробел вместо отсутствующего reasoning. Это не подтверждение поведения установленного шлюза. Перед выпуском зафиксировать его версию/config и проверить round-trip на синтетическом двухшаговом tool-сценарии; потерю native effort исправлять на gateway/passthrough, не маскировать интерфейсом. 

Приоритет:

```text
MessageCreate.reasoning_effort
→ session.configuration.reasoning_effort
→ model.default_effort
```

Override сохраняется на user Message и действует до следующего реального user Message, включая approval resume. `auto` — политика, никогда wire-значение.

```python
async def decide(self, context: ReasoningContext) -> ReasoningDecision
def adapt_reasoning(model: ResolvedModel, decision: ReasoningDecision,
                    max_tokens: int) -> dict[str, Any]
```

Эвристика v1: ask→low, dev/прочие→medium; задача ≥8000 символов→high; ошибка предыдущего шага повышает уровень на один, две последовательные→max. Выбирать только доступные уровни; сохранять requested/effective/source/reason. Это интерфейс будущего Jev; сейчас никаких Jev-вызовов.

Расширить `ReasoningSpec`: adapter, effort mapping, token budgets, history policy. `allowed_efforts` сохранить; `auto` принимать отдельно. Несовместимый explicit override → `422`, не молчаливое игнорирование.

- OpenAI effort: проверенное native значение.
- DeepSeek: `thinking.type` + реально поддерживаемый effort; не Anthropic budget.
- Anthropic manual: `thinking={type:enabled,budget_tokens:...}`, конфигурационные бюджеты 1024/4096/8192/16384, ограниченные output budget. Только для поддерживающих manual thinking моделей; signed blocks round-trip неизменными, не реконструировать из текста. 
- `enable_thinking`: Boolean по объявленным возможностям gateway.
- Unsupported reasoning: параметров нет; children/internal summary default low независимо от родителя.

Wire-адаптер через LiteLLM **не означает** поддержку нативного `kind=anthropic`/Responses transport.

## 5. Контекст и инкрементальная история

Точки: `src/agent/context.py:25`, `src/agent/context.py:30`, `src/agent/runtime.py:264`, `src/agent/runtime.py:309`, `src/agent/api.py:208`.

```python
async def load_page(session_id: str, after_id: int, limit: int = 200) -> HistoryPage
async def prepare(history: HistoryView, model: ResolvedModel,
                  tools: list[dict[str, Any]], target_tokens: int | None) -> ContextResult
async def calibrate(model_key: str, estimate: int, prompt_tokens: int) -> None
```

Оценивать **wire-проекцию**: messages, reasoning, tool definitions, protocol overhead. Коэффициент на provider/model/wire-version: первоначально `1`; EWMA `0.8*old + 0.2*(actual/raw)`, clamp `[0.5,4]`; только известный положительный prompt usage, один sample на attempt. Хранить коэффициент/samples в БД.

Input budget:

```text
context_window - max(reserved_tokens, effective_max_tokens) - safety_margin
safety_margin = max(256, 2% context_window)
```

После повышения output budget пересчитать контекст.

Неделимый блок — assistant.tool_calls + все tool-ответы + reasoning. Защитить system/instructions, исходную задачу, последний user, незавершённый блок и два последних завершённых блока.

Ступени, только в проекции:

1. Маскировать старые tool-результаты структурированным JSON: success/error, path, размеры/hash, краткий excerpt. Не резать JSON строкой.
2. Сворачивать большие content/old_text/new_text аргументы старых write/edit, сохраняя JSON, paths/hash и связь с результатом. Такие проекции никогда не исполняются.
3. Суммаризировать старый завершённый prefix моделью `routing.light`; добавить этот routing field с проверкой reference, не угадывать по имени модели. Сохранить `Message(kind=summary)`, coverage, source hash, version. В wire помещать как явно обозначенные данные до актуального user, не как новые привилегированные инструкции.

Summary содержит цель, ограничения, решения, изменённые пути, результаты проверок и незакрытые задачи. Ограничить output 1024 tokens и число summary-вызовов четырьмя; каждый учитывается/audit-ится. Исходники остаются неизменными. Ошибка summary → deterministic compression завершённых блоков, без бесконечного recursion.

`LLMError` получает typed kind/status/provider code. Context overflow распознавать по структурированному коду и allowlist известных сообщений, **не по любому HTTP 400**. LiteLLM различает context-window exception. 

После overflow: target `min(input_budget, 0.60*context_window)`, сжатие и максимум **два** повтора. Если защищённая часть/tools уже больше target либо размер не уменьшается — `ContextBudgetError`, без loop.

На execution snapshot брать `effective_registry`; не перечитывать YAML/overrides каждый шаг. Историю cold-start загружать summary checkpoint + protected/tail pages; далее только `id > cursor`, с ограниченным cache. Unfinished calls выбирать отдельным индексированным запросом. `/context` использует ту же оценку, но не запускает платную суммаризацию.

## 6. Дочерние агенты

Точки: `src/agent/runtime.py:639`, `src/agent/tools/advanced.py:382`, `src/agent/tools/base.py:22`.

```python
async def spawn_child(fence: RunFence, tool_call_id: str,
                      request: ChildRequest) -> ChildStarted
async def resolve_child_wait(db: AsyncSession, child_id: str,
                             child_generation: int) -> None
```

Добавить typed результат `ToolOutcome | SuspendChild`, не использовать `response:null` как завершение.

Транзакция spawn: проверить parent/approval/generation → создать ребёнка, уникальную связь по parent tool-call → enqueue ребёнка → при wait=true перевести call и parent в `waiting_child`. Повтор spawn возвращает существующую связь. `_run` возвращается; semaphore и lease освобождаются. Ребёнок проходит обычный worker/guarded путь.

Approvals ребёнка сохраняют ожидание родителя. Terminal ребёнка атомарно создаёт durable child-done intent; обработчик новой транзакцией проверяет parent generation, фиксирует результат ровно один раз и пробуждает родителя. Захватывать результат **конкретного child generation**, не последний assistant из всей сессии.

wait=false сразу завершает tool результатом с child ID. Persisted deadline использует `max_child_wait_seconds` как wall-clock, включая approvals; reaper останавливает просроченного ребёнка и возвращает timeout родителю. Stop родителя отменяет descendants/links; позднее завершение не воскрешает его. Redis Pub/Sub не является единственным completion-сигналом.

## 7. Tools, usage и fallback

**Параллельность:** `src/agent/runtime.py:454`, `src/agent/runtime.py:506`, `src/agent/tools/registry.py:71`.

Добавить явный `parallel_safe=False`; разрешить только проверенным read-only tools. Выполнять последовательные read-only сегменты через `gather`, общий worker semaphore `max_parallel_tools=4`; mutation/child calls — barriers. Каждая задача имеет собственную DB session. Ошибка одного tool не теряет остальные результаты; Stop отменяет и ожидает весь batch.

Сохранять `assistant_message_id/ordinal` у calls. Результаты фиксировать независимо; tool Messages материализовать координатором после закрытия группы в исходном порядке, с unique tool-call constraint. Recovery не создаёт дублей и не запускает LLM до всех ответов.

**Usage:** `src/agent/model_config.py:115`, `src/agent/models.py:104`.

Ledger на каждый HTTP attempt: actual model/provider/config, purpose, nullable prompt/completion/reasoning/cache usage, pricing snapshot, Decimal cost, status/timestamps. Стоимость — `(input*input_rate + completion*output_rate)/1_000_000`; reasoning второй раз не прибавлять. Учитывать retries, summaries и fallback. Missing usage/pricing → unknown, не ноль; без cached rates сумма является оценкой. `/sessions/{id}/usage` разделяет собственные расходы и descendants, без двойного счёта.

**Fallback:** `src/agent/model_config.py:137`, `src/agent/model_registry.py:42`, `src/agent/llm.py:101`.

После transient retries идти по `routing.fallback` без циклов. Retry-After seconds/date + jitter; общий cap **12 HTTP attempts на inference**, отдельный context cap 2. Auth, validation, missing reasoning и schema errors не переключают модель. Каждый кандидат имеет собственные context/reasoning/audit/pricing; несовместимый history/tool transport пропускается.

Circuit breaker на provider: пять исчерпанных transport/5xx failures → open 30 с → один half-open probe с lease. 429 соблюдает cooldown, но не выключает весь provider автоматически. Redis workers используют общий breaker state. Профиль сессии не переписывается. In-flight клиентов старого snapshot не закрывать при reload.

## 8. Миграции, шаги и приёмка

После текущей `0006` — согласовать следующие свободные revision IDs:

- **Runtime/outbox:** Session generation/wake/served/event sequence, owner epoch/token/lease; Event sequence; outbox с UUID, kind, payload, available/claimed/delivered timestamps и retries; unique enqueue `(session,generation,wake_seq)`.
- **Messages/attempts:** reasoning TEXT, kind/status, requested effort, generation, attempt FK, provider metadata, summary coverage/hash/version; attempt ledger; calibration; составные индексы `(session_id,id)` и `(session_id,status)`.
- **Children/tools:** waiting_child значения, child_waits с generations/deadline/result, unique parent-call; call assistant FK/ordinal; unique tool Message.

Backfill: старые Messages complete/normal, reasoning null; поколения 0. Старые call grouping восстанавливать только однозначно. Upgrade проверять SQLite/PostgreSQL; предыдущие миграции не редактировать.

**Порядок маленьких PR:** schema/fences → outbox+wake/ACK → watchdog/reclaim → child waits → SSE parser → broker/WS → reasoning adapters → context stages → parallel tools/usage → fallback/breaker → fault acceptance/rollout.

**Обязательные тесты:**

- Unit: SSE fragment/interleaving/null/usage/error/EOF; reset между attempts; exact audit bytes; reasoning priority/auto/mapping; calibration и budget с tools; JSON compression, summary coverage и ≤2 overflow retries; gather order/barriers.
- Fakeredis: generation-aware dedupe; approval-before-ACK с barriers; Lua ownership; пустая reclaim page с ненулевым cursor; cancel loss/reconnect; breaker half-open.
- Реальные PostgreSQL+Redis, два workers: duplicate claim, epoch loss, shutdown cleanup, crash до/после XADD и dispatch отметки, Redis restart/reconciliation, Stop/approval/result races, поздний старый ACK.
- Child integration: wait=true, approval/rejection, timeout, Stop parent, duplicate child-done, рестарт между terminal и parent resume; `max_parallel_sessions=1` без deadlock.
- Gateway opt-in: V4/V3.2 reasoning→tools→continuation, native effort сохранён, Anthropic blocks round-trip; только синтетические данные.
- Performance: нет повторной загрузки всей истории; количество Event rows не зависит от числа дельт; usage включает каждую оплачиваемую попытку.

**Стыки с уже сделанным:** B1 — внешний wire ID и ordinal нужны SSE/replay; B6 — незавершённый stream нельзя принять за stop, увеличение max_tokens требует нового budget; B22 — intake и execution должны использовать один schema snapshot.

Главные риски: неподтверждённый gateway mapping, стоимость reasoning/retries, нетранзакционные side effects, mixed-version rollout и действующий UI, который пока refetch-ит на каждое WS-событие. Полную фазу UI сюда не включать, но минимальный consumer дельт нужен для полезного стриминга.
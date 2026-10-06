# Шаг 2.3 — интеграция управления контекстом

Дата: 2026-10-06. Ветка: `dev`. Основа — раздел 5 `phase-2-spec.md`.

## 1. Область изменений

Подключить сжатие контекста, калибровку оценки токенов, checkpoint суммаризации и обработку переполнения.

Сохранить исходные сообщения. Менять только проекцию запроса.

Не включать в этот шаг дерево сообщений, редактирование истории, полный журнал расходов, circuit breaker и generation/lease. Эти механизмы относятся к следующим шагам: `docs/review/phase-2-progress.md:35`.

### 1.1. Точки интеграции

| Точка | Изменение |
|---|---|
| `src/agent/context.py:19` | Заменить обрезку сообщений последовательностью ступеней |
| `src/agent/context_compaction.py:42` | Использовать структурное маскирование результатов |
| `src/agent/context_compaction.py:92` | Использовать свёртку аргументов |
| `src/agent/token_calibration.py:28` | Подключить наблюдения фактического usage |
| `src/agent/runtime.py:383` | Готовить контекст после выбора модели, reasoning и tools |
| `src/agent/runtime.py:662` | Пересчитывать контекст при изменении output budget |
| `src/agent/runtime.py:720` | Провести fallback через тот же путь подготовки |
| `src/agent/api.py:209` | Подключить бесплатный preview контекста |
| `src/agent/llm.py:18` | Добавить структурированные ошибки |
| `src/agent/llm.py:82` | Выделить общий построитель wire-запроса |
| `src/agent/models.py:93` | Добавить метаданные checkpoint |
| `migrations/versions/0007_message_reasoning.py:12` | Использовать `0007` как предыдущую миграцию |

Интерфейсы чистых модулей `context_blocks`, `llm_errors`, `context_budget` и `context_summary` — входной контракт этого документа.

### 1.2. Инварианты

- Не менять `Message.content`, `tool_calls`, reasoning и результаты инструментов ради сжатия.
- Не изменять защищённые блоки.
- Не разрывать exchange.
- Не отправлять определения tools вне оценки бюджета.
- Не исполнять аргументы из сжатой проекции.
- Не считать summary новым запросом пользователя.
- Не повторять запрос после первой содержательной дельты.
- Не выполнять сетевое ожидание внутри открытой транзакции.

---

## 2. Контракты подготовки контекста

### 2.1. Общий wire-запрос

Выделить один построитель запроса. Использовать его для оценки, отправки и audit.

```python
@dataclass(frozen=True, slots=True)
class WireRequest:
    payload: dict[str, Any]
    body: bytes
    wire_version: str


def build_wire_request(
    model: ResolvedModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    reasoning_effort: str | None,
    max_tokens: int,
    stream: bool,
) -> WireRequest: ...


def estimate_wire(request: WireRequest) -> int: ...
```

`body` содержит окончательные байты JSON. После построения не менять `payload`.

Отправлять именно `body`. Хешировать именно `body`.

Включить `stream` и `stream_options` до сериализации. Сейчас streaming-параметры добавляются позже: `src/agent/llm.py:161`.

### 2.2. История

```python
@dataclass(frozen=True, slots=True)
class HistoryPage:
    messages: tuple[Message, ...]
    cursor: int
    has_more: bool


class HistoryView(Protocol):
    session_id: str
    messages: Sequence[Message]
    checkpoint: SummaryCheckpoint | None
    checkpoint_message_id: int | None
    checkpoint_model_id: str | None
    covered_messages: int
    cursor: int

    async def source_prefix(
        self,
        covers_until: int,
    ) -> Sequence[Message]: ...


async def load_page(
    session_id: str,
    after_id: int,
    limit: int = 200,
) -> HistoryPage: ...
```

`messages` содержит исходные сообщения, доступные текущему контексту. При наличии checkpoint это начальные защищённые сообщения и хвост после coverage.

`source_prefix()` загружает исходный prefix страницами. Вызывать его при создании или расширении checkpoint. Не вызывать его на каждом inference-шаге.

Каждая операция загрузки закрывает свою транзакцию до возврата.

### 2.3. Суммаризация

```python
@dataclass(frozen=True, slots=True)
class SummaryOutcome:
    checkpoint: SummaryCheckpoint
    model_id: str


class SummaryRunner(Protocol):
    model: ResolvedModel

    async def __call__(
        self,
        items: list[dict[str, Any]],
        previous: SummaryCheckpoint | None,
        *,
        covers_until: int,
        source_hash: str,
    ) -> SummaryOutcome | None: ...
```

`SummaryRunner` выполняет сетевой вызов. `ContextManager` выбирает покрываемые блоки.

`None` означает отказ от платной суммаризации. После него перейти к детерминированной свёртке.

### 2.4. Результат

```python
CompactionStage = Literal[
    "none",
    "stage_1",
    "stage_2",
    "summary",
    "deterministic",
]


@dataclass(frozen=True, slots=True)
class ContextResult:
    messages: list[dict[str, Any]]
    request: WireRequest
    budget: Budget
    target_tokens: int
    effective_max_tokens: int
    raw_tokens: int
    approximate_tokens: int
    tokens_before: int
    calibration_factor: float
    calibration_samples: int
    stage: CompactionStage
    fits: bool
    omitted_messages: int
    masked_tool_results: int
    folded_tool_calls: int
    summary_used: bool
    summary_calls: int
    checkpoint_message_id: int | None
    checkpoint_to_save: SummaryCheckpoint | None
    checkpoint_model_id: str | None
    would_summarize: bool

    @property
    def truncated_tool_results(self) -> int: ...


class ContextBudgetError(RuntimeError):
    def __init__(
        self,
        reason: str,
        result: ContextResult,
    ) -> None: ...
```

Сохранить `truncated_tool_results` как совместимый alias для `masked_tool_results`. Новая реализация не режет результат инструмента строкой.

`omitted_messages` считает исходные видимые сообщения, заменённые checkpoint. Не включать в него `skipped`, `interrupted` и строки summary.

`tokens_before` описывает проекцию до текущего сжатия. При переиспользовании checkpoint эта проекция уже содержит checkpoint, а не полный архив.

### 2.5. `ContextManager`

```python
class ContextManager:
    def __init__(
        self,
        reserved_tokens: int,
        calibrator: TokenCalibrator,
    ) -> None: ...

    async def prepare(
        self,
        history: HistoryView,
        model: ResolvedModel,
        tools: list[dict[str, Any]],
        *,
        effective_max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        target_tokens: int | None = None,
        summary_runner: SummaryRunner | None = None,
        stream: bool = True,
    ) -> ContextResult: ...
```

Заменить фабрику координатора:

```python
def context_for(
    self,
    calibrator: TokenCalibrator,
) -> ContextManager: ...
```

Передавать модель в `prepare()`. Не хранить бюджет конкретной модели в долгоживущем менеджере.

### 2.6. Бюджет

Использовать только `context_budget.input_budget()`.

```text
effective_max_tokens = переданный лимит или model.max_tokens
output_tokens = max(reserved_tokens, effective_max_tokens)
safety_margin = max(256, ceil(context_window * 0.02))
input_tokens = context_window - output_tokens - safety_margin
target = min(input_tokens, target_tokens), если target_tokens задан
target = input_tokens, иначе
```

`Budget.output_tokens` означает резерв для ответа. Фактический `max_tokens` хранить отдельно в `effective_max_tokens`.

Не применять нижнюю границу `1024` к input budget. Текущая граница может скрыть невозможный запрос: `src/agent/context.py:23`.

Если `input_tokens <= 0` или `target <= 0`, поднять `ContextBudgetError(reason="invalid_budget")`.

### 2.7. Оценка wire-проекции

Для первой версии зафиксировать воспроизводимую эвристику.

1. Взять окончательный `payload`.
2. Выделить поля `messages`, `tools`, `tool_choice` и параметры reasoning.
3. Исключить `model`, `max_tokens`, `stream` и `stream_options`.
4. Сериализовать проекцию через `ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`.
5. Посчитать количество символов JSON.
6. Добавить protocol overhead.

```text
raw_tokens =
    ceil(json_chars / 4)
    + 8
    + 4 * len(messages)
    + 8 * len(tools)
```

Считать аргументы tool calls как строки внутри wire JSON. Учитывать их экранирование.

Считать только reasoning, который реально попадает в запрос.

Установить версию оценщика `chat-json-v1`. Включить fingerprint конечного endpoint в `wire_version`. Смена endpoint не должна наследовать старую калибровку.

```text
wire_version = "chat-json-v1:" + sha256(normalized_endpoint)
```

Изменение алгоритма оценки, состава prompt-полей или wire-адаптера требует новой версии.

Применять калибровку один раз к полной оценке:

```python
approximate_tokens = calibrator.adjust(key, raw_tokens)
```

Не применять `adjust()` отдельно к каждому сообщению.

Зафиксировать factor на время одного `prepare()`. Сравнивать все ступени с одним factor.

Оценивать защищённый минимум отдельным полным wire-запросом:

```text
защищённые сообщения + tools + reasoning-параметры
```

Не складывать независимые оценки защищённых сообщений и tools. Это повторно посчитает overhead.

### 2.8. Блоки и защита

Использовать `split_blocks()` и `protected_blocks()`.

Принять границы `Block` как полуинтервал `[start, end)`. `start` и `end` — позиции в списке, не `Message.id`.

Защитить:

- Начальные `system`.
- Первый настоящий `user`.
- Последний настоящий `user`.
- Незавершённые exchange.
- Два последних завершённых блока.

Определять защиту до вставки summary.

Если встречается дополнительный `system`, сохранить его как отдельную защищённую границу. Не переносить его в summary.

Использовать `mask_tool_result()` и `fold_write_arguments()` адресно. Не использовать текущий `compact()` как управляющий алгоритм: он защищает последние шесть сообщений, а не exchange-блоки — `src/agent/context_compaction.py:136`.

### 2.9. Порядок ступеней

Остановиться на первой проекции, которая проходит `fits()`.

| Ступень | Действие |
|---|---|
| Без сжатия | Построить исходную проекцию или проекцию с существующим checkpoint |
| Ступень 1 | Маскировать старые tool-результаты в незашищённых завершённых блоках |
| Ступень 2 | Дополнительно свернуть большие аргументы старых `write_file` и `edit_file` |
| Summary | Заменить старый завершённый prefix результатом `SummaryRunner` |
| Детерминированная свёртка | Заменить тот же prefix результатом `deterministic_summary()` |
| Ошибка | Поднять `ContextBudgetError` |

Ступень 2 строить из исходной проекции. Применять маскирование и свёртку один раз. Не маскировать повторно уже маскированный JSON.

Сохранять преобразование только при уменьшении полной raw-оценки. Маленький результат может вырасти после маскирования.

Перед платной ступенью проверить защищённый минимум. Если он превышает target, сразу поднять `protected_over_budget`.

Для summary выбирать последовательный prefix завершённых незашищённых блоков после исходной задачи. Остановиться перед первым защищённым блоком.

Не удалять произвольные блоки из середины истории.

---

## 3. Хранение и миграция `0008`

Создать `migrations/versions/0008_context_state.py`.

```text
revision = "0008"
down_revision = "0007"
```

Не добавлять `messages.kind` повторно. Поле уже добавлено миграцией `0007`: `migrations/versions/0007_message_reasoning.py:19`.

### 3.1. Таблица `token_calibrations`

| Поле | Тип | Ограничения |
|---|---|---|
| `provider` | `TEXT` | `NOT NULL`, часть PK |
| `model` | `TEXT` | `NOT NULL`, часть PK |
| `wire_version` | `VARCHAR(96)` | `NOT NULL`, часть PK |
| `factor` | `DOUBLE PRECISION` / `Float(53)` | `NOT NULL`, default `1.0` |
| `samples` | `INTEGER` | `NOT NULL`, default `0` |
| `updated_at` | `DateTime(timezone=True)` | `NOT NULL`, UTC |

Добавить ограничения:

```text
pk_token_calibrations(provider, model, wire_version)
ck_token_calibrations_factor: factor BETWEEN 0.5 AND 4.0
ck_token_calibrations_samples: samples >= 0
```

Отдельный индекс на ключ не нужен. Его обеспечивает PK.

Использовать:

```text
provider = ResolvedModel.provider_id
model = ResolvedModel.model
```

Не использовать только профиль `ResolvedModel.id`. Несколько профилей могут обращаться к одной модели.

Для `TokenCalibrator` кодировать составной ключ JSON-массивом. Не соединять части разделителем.

### 3.2. Поля `Message`

Добавить четыре nullable-поля:

| Поле | Тип | Значение для summary |
|---|---|---|
| `covers_until` | `INTEGER` | Последний покрытый исходный `Message.id`, включительно |
| `source_hash` | `VARCHAR(64)` | Хеш исходного prefix |
| `version` | `INTEGER` | Версия формата checkpoint и хеширования |
| `model` | `VARCHAR(100)` | ID профиля, создавшего summary |

Для детерминированного checkpoint установить `model=NULL`.

Использовать `version=1`. Это версия формата, не порядковый номер summary.

Записывать checkpoint так:

```text
kind = "summary"
role = "user"
content = нормализованный текст summary
tool_calls = NULL
tool_call_id = NULL
reasoning_content = NULL
reasoning_effort = NULL
skipped = false
```

Маркер «данные, не инструкции» добавляет `render_summary_item()`. Не добавлять его повторно при сохранении.

Для обычных и прерванных сообщений оставить четыре новых поля `NULL`.

`covers_until` не является FK. Это граница coverage. При сохранении проверить принадлежность границы той же сессии.

### 3.3. Индексы и ограничения

Добавить:

```text
ix_messages_session_id_id
    (session_id, id)

ix_messages_session_kind_coverage
    (session_id, kind, covers_until, id)

uq_messages_summary_checkpoint
    UNIQUE(session_id, covers_until, source_hash, version)

ix_tool_calls_session_status
    (session_id, status)
```

Сохранить существующий `ix_messages_session_id`: `migrations/versions/0004_indexes_drop_commands.py:20`.

Nullable-поля уникального ключа позволяют хранить обычные сообщения без ограничения их количества.

Добавить именованное CHECK-ограничение:

- Для `kind="summary"` требовать `role="user"`, непустой `content`, положительные `covers_until` и `version`, ненулевой `source_hash`.
- Для остальных kind требовать `NULL` во всех четырёх новых полях.

Проверять длину и формат SHA-256 в коде. Не вводить диалектозависимую проверку hex-строки.

### 3.4. Upgrade

1. Создать таблицу калибровки.
2. Добавить четыре nullable-поля через `batch_alter_table`.
3. Добавить CHECK-ограничение.
4. Добавить индексы.
5. Добавить уникальный ключ.

Не изменять содержимое существующих сообщений.

Проверять миграцию на SQLite и PostgreSQL. Batch-режим позволяет использовать один путь миграции для обеих БД. ([alembic.sqlalchemy.org](https://alembic.sqlalchemy.org/en/latest/batch.html))

### 3.5. Downgrade

1. Удалить строки `kind="summary"`.
2. Удалить новые индексы и уникальный ключ.
3. Удалить CHECK-ограничение.
4. Удалить четыре поля через `batch_alter_table`.
5. Удалить `token_calibrations`.

Удалять только производные checkpoint. Сохранять обычные и прерванные сообщения.

Удалять именованное CHECK-ограничение до удаления его колонок. ([alembic.sqlalchemy.org](https://alembic.sqlalchemy.org/en/latest/batch.html))

### 3.6. Обновление калибровки

```python
async def calibrate(
    key: CalibrationKey,
    raw_estimate: int,
    prompt_tokens: int | None,
    *,
    audit_id: int,
) -> None: ...
```

Использовать текущую формулу: `src/agent/token_calibration.py:35`.

```text
factor = clamp(
    0.8 * old_factor + 0.2 * prompt_tokens / raw_estimate,
    0.5,
    4.0,
)
samples = samples + 1
```

При `raw_estimate <= 0`, неизвестном usage или `prompt_tokens <= 0` не менять factor и samples.

Обновлять строку через CAS по `samples`. При конфликте перечитать factor и повторить расчёт. Не сохранять общий snapshot по принципу «последняя запись побеждает».

Фиксировать terminal audit и калибровку одной короткой транзакцией. Условный переход audit из `started` в terminal допускает только одно наблюдение для HTTP attempt.

Загружать factor перед подготовкой запроса. После успешного commit обновлять локальный cache.

---

## 4. `routing.light`

Оставить имя `light`.

```python
class RoutingSpec(_Spec):
    default: str
    fallback: list[str]
    roles: dict[str, str]
    light: str | None = None
```

Зафиксировать поведение:

- `light` задан → выбрать указанную модель.
- `light=None` → выбрать `routing.default`.
- Выбранная модель не настроена → пропустить платную ступень.
- Суммаризация завершилась ошибкой → перейти к детерминированной свёртке.
- Не переключать summary по `routing.fallback`.
- Не угадывать модель по имени профиля.

Это соответствует текущему методу `ResolvedModelRegistry.light()`: `src/agent/model_config.py:283`.

Проверять reference и поддерживаемый transport в `ModelsConfig.validate_references()`: `src/agent/model_config.py:199`.

Проводить такую же проверку после применения DB overrides.

Передавать `light_id` через все пути построения registry. Дополнить обратное преобразование registry в конфигурацию: сейчас оно перечисляет только `default`, `fallback` и `roles` — `src/agent/model_overrides.py:203`.

Для API настроек:

- Отсутствующее поле PATCH сохраняет текущее значение.
- `"light": null` сбрасывает значение к `routing.default`.
- Неизвестная или отключённая модель возвращает `422`.

Использовать один effective registry snapshot на выполнение `_run()`. Текущий runtime уже создаёт такой snapshot: `src/agent/runtime.py:326`.

---

## 5. Поток выполнения в `runtime.py`

### 5.1. Состояние логического inference

Логический inference — получение одного ответа модели до его сохранения. Он включает подготовку, summary, transient retries, overflow retries и fallback.

```python
@dataclass(slots=True)
class InferenceState:
    inference_id: str
    http_attempts: int = 0
    summary_calls: int = 0
    overflow_retries: int = 0
    started: bool = False
    visited_models: set[str] = field(default_factory=set)
    sent_requests: set[str] = field(default_factory=set)
```

Установить лимиты:

```text
MAX_HTTP_ATTEMPTS = 12
MAX_SUMMARY_CALLS = 4
MAX_CONTEXT_RETRIES = 2
```

Не сбрасывать счётчики при fallback или переподготовке контекста.

Сбросить их только перед следующим логическим inference.

Каждый фактический HTTP-запрос расходует `http_attempts`. Summary дополнительно расходует `summary_calls`.

### 5.2. Основной порядок

1. Загрузить состояние сессии.
2. Завершить или дождаться незавершённых tool calls.
3. Получить registry snapshot.
4. Выбрать модель.
5. Выбрать reasoning.
6. Получить определения tools.
7. Загрузить калибровку.
8. Вызвать `prepare()`.
9. Сохранить новый checkpoint.
10. Опубликовать статистику сжатия.
11. Отправить подготовленный wire-запрос.
12. Зафиксировать usage и калибровку.
13. Сохранить ответ с параметрами фактически использованной модели.

Перенести получение tools перед `prepare()`. Сейчас tools строятся после подготовки: `src/agent/runtime.py:397`.

Исключить строки summary из выбора reasoning. Сейчас `_reasoning_for()` выбирает все строки с `role="user"`: `src/agent/runtime.py:765`.

Исключить summary из проверки последнего обычного сообщения и завершённости шага: `src/agent/runtime.py:345`.

### 5.3. Асинхронная суммаризация

Запускать summary только после неуспеха ступеней 1–2.

Использовать `await SummaryRunner(...)`. Не создавать фоновую задачу, которая продолжит работу после Stop.

Для summary:

```text
tools = []
stream = false
reasoning_history = "none"
max_tokens = min(SUMMARY_MAX_TOKENS, light.max_tokens)
```

Выбирать `low`, если модель поддерживает этот уровень. Иначе выбирать минимальный объявленный уровень. Не наследовать override сессии.

Строить запрос через `build_summary_request(items, previous)`.

Подбирать наибольшую последовательную группу целых блоков, которая помещается в собственный бюджет light-модели.

Если запрос summary слишком велик:

1. Применить ступень 1 к его исходным данным.
2. Применить ступень 2 к его исходным данным.
3. Уменьшить группу на границе блока.
4. Отказаться от платной ступени, если не помещается даже один блок.

Не суммаризировать запрос суммаризации. Не вызывать `prepare()` рекурсивно с активным `SummaryRunner`.

Не выполнять скрытые transient retries для summary. Один ошибочный вызов переводит подготовку к детерминированной ступени.

Следующий summary-вызов допустим только при увеличении coverage и уменьшении raw-оценки основной проекции.

Остановиться после четырёх HTTP-вызовов summary.

Считать результат непригодным, если:

- Текст пуст после `normalize_summary()`.
- Ответ содержит tool calls.
- Ответ обрезан по `finish_reason="length"`.
- Ответ не уменьшает проекцию.
- Результат нарушает лимит output.

Ошибка summary не является ошибкой всей сессии, пока детерминированная ступень может уложить контекст.

`CancelledError` не превращать в детерминированную свёртку. Зафиксировать отмену audit и передать отмену выше.

### 5.4. Coverage и source hash

`covers_until` — максимальный исходный ID заменённого prefix.

Coverage не означает удаление защищённых сообщений с меньшими ID. Начальные `system` и первая задача остаются в wire.

Считать `source_hash()` по исходным сообщениям до `covers_until` включительно. Не хешировать сжатую проекцию.

Использовать стабильное представление:

```text
id, role, kind, skipped, content,
tool_calls, tool_call_id,
reasoning_content, reasoning_effort
```

Не включать timestamps и строки summary.

Применять одну и ту же функцию `source_hash()` при создании и проверке checkpoint. Метаданные источника не отправлять в wire.

При расширении checkpoint:

1. Получить новый завершённый prefix.
2. Передать предыдущий checkpoint в `build_summary_request()`.
3. Пересчитать source hash по полному исходному coverage.
4. Создать checkpoint с новой границей.
5. Оставить предыдущий checkpoint в БД.

Не выводить новый source hash из текста предыдущего summary.

### 5.5. Сохранение checkpoint

Сохранять только checkpoint, участвующий в итоговой пригодной проекции.

Перед сохранением:

1. Перечитать статус сессии.
2. Проверить отсутствие Stop.
3. Проверить отсутствие изменений исходного snapshot.
4. Проверить границу завершённого блока.
5. Проверить метаданные checkpoint.

Затем вставить `Message(kind="summary")` короткой транзакцией.

При конфликте уникального ключа использовать существующую строку.

При изменении истории отбросить кандидат и повторить локальную подготовку. Не оплачивать повторный summary в том же цикле автоматически.

Не сохранять checkpoint из `/context`.

### 5.6. Переиспользование checkpoint

Выбирать checkpoint с наибольшим допустимым `covers_until`. При равной границе выбирать последнюю пригодную строку.

Пригодный checkpoint имеет:

- Поддерживаемую `version`.
- Непустой текст.
- Корректный source hash.
- Границу завершённого prefix.
- Непересекающуюся защищённую незавершённую часть.

Строить wire:

```text
начальные system
→ первая настоящая user-задача
→ render_summary_item(checkpoint)
→ исходный хвост с id > covers_until
```

Сохранять относительный порядок сообщений. Не дублировать первую задачу, если она входит в хвост.

Не добавлять остальные строки `kind="summary"` как обычные сообщения.

При cold start загружать checkpoint, начальные защищённые сообщения и хвост страницами. После этого загружать только новые исходные сообщения по `id > cursor`.

Запрашивать незавершённые calls отдельно. Не загружать все calls через relationship на каждом шаге.

Шаг 2.3 использует append-only историю. Проверять source hash при создании и расширении checkpoint. Будущие операции edit, skip и смены ветки обязаны инвалидировать checkpoint.

### 5.7. Детерминированная свёртка

Использовать `deterministic_summary()` для того же завершённого prefix.

При наличии предыдущего checkpoint включить его данные в свёртку. Не терять ранее покрытую историю.

Ограничить текст локальной оценкой до `SUMMARY_MAX_TOKENS`. При необходимости удалять целые записи в фиксированном порядке. Не резать сериализованный JSON.

После вставки снова оценить весь wire-запрос.

Если проекция подходит, вернуть checkpoint с `model=NULL`.

Если проекция не подходит, поднять `ContextBudgetError`. Не удалять защищённые данные ради успешного ответа.

### 5.8. Typed `LLMError`

```python
class LLMError(RuntimeError):
    def __init__(
        self,
        info: ProviderErrorInfo,
        *,
        started: bool = False,
    ) -> None: ...


class LLMTransientError(LLMError):
    pass
```

Сохранить `LLMTransientError` для совместимости. Принимать решение по `info.kind`, а не по тексту исключения.

Использовать один классификатор для:

- Обычного HTTP-ответа.
- HTTP-ошибки при streaming.
- Ошибки внутри успешного SSE-потока.
- Timeout и network exception.

Сейчас SSE-ошибка теряет структурированный code: `src/agent/sse.py:106`.

Классифицировать полный допустимый error body до сокращения текста. После классификации применить `redact()` к сохраняемому `message`.

Overflow распознавать только через `is_context_overflow(info)`.

Не считать произвольный `400`, `bad_request` или `finish_reason="length"` переполнением контекста.

Overflow не повторяет транспортный retry-loop. Его обрабатывает координатор контекста.

### 5.9. Повторы после overflow

Для каждого overflow:

1. Проверить `started=False`.
2. Проверить `overflow_retries < 2`.
3. Рассчитать бюджет текущей модели.
4. Рассчитать target.
5. Повторно вызвать `prepare()`.
6. Проверить уменьшение raw-оценки.
7. Проверить новый fingerprint запроса.
8. Увеличить `overflow_retries`.
9. Отправить запрос.

Базовое ограничение:

```text
overflow_cap = overflow_target(budget)
             = min(input_tokens, floor(context_window * 0.60))
```

Для принудительного прогресса:

```text
retry_target = min(
    overflow_cap,
    floor(previous_raw_adjusted_estimate * 0.80),
    previous_target,
)
```

`previous_raw_adjusted_estimate` — оценка предыдущего raw-размера с factor текущей подготовки. Это исключает ложный прогресс из-за смены factor.

Первый запрос и два повтора дают максимум три отправки по overflow-пути.

Не повторять запрос, если:

- Защищённый минимум превышает target.
- Новая raw-оценка не меньше предыдущей.
- Fingerprint уже отправлялся.
- Исчерпан общий HTTP cap.
- Исчерпаны два overflow-повтора.

Использовать причины `protected_over_budget`, `no_progress` и `retry_exhausted`.

Не переключать модель из-за самого overflow. При исчерпании сжатия вернуть `ContextBudgetError`.

### 5.10. Fallback и streaming

Проводить primary и fallback через одну функцию inference.

```python
@dataclass(frozen=True, slots=True)
class InferenceResult:
    response: LLMResponse
    model: ResolvedModel
    reasoning: ReasoningDecision
    context: ContextResult
```

Для каждого fallback заново выбирать:

- Бюджет.
- Калибровку.
- Reasoning.
- Политику reasoning history.
- Wire-запрос.
- Audit destination.

Не использовать текущий fallback-вызов `prepare(session.messages)` без reasoning policy: `src/agent/runtime.py:734`.

Возвращать фактическую модель и reasoning вместе с ответом. Не сохранять параметры primary для ответа fallback.

Переключать модель только после исчерпания transient retries для `rate_limit`, `server`, `timeout` или `network`.

Не переключать модель для `auth`, `not_found`, `bad_request`, ошибок схемы и `ContextBudgetError`.

Использовать `visited_models`. Не заходить в одну fallback-модель повторно.

Установить `started=True` при первой непустой дельте:

- `content`.
- Reasoning.
- Tool call или его arguments.

Пустой chunk, heartbeat и role-only chunk не устанавливают `started`.

После `started=True` запретить transport retry, overflow retry, fallback и автоматическое увеличение output budget.

Текущий `_complete()` повторяет обрезанный ответ после streaming: `src/agent/runtime.py:711`. Для streaming этот повтор нужно отключить.

Если output budget увеличивается до первой дельты, заново вызвать `prepare()` с новым `effective_max_tokens`.

Не очищать уже полученный текст через `stream.reset` ради повторной генерации. Сохранение частичного ответа остаётся отдельным шагом.

### 5.11. Audit каждого HTTP attempt

Перенести audit на границу фактической HTTP-отправки. Текущая граница не видит внутренние retries клиента: `src/agent/llm.py:129`.

Создавать одну audit-строку со статусом `started`. Обновлять эту же строку до `completed`, `error` или `cancelled`.

Перед отправкой записывать:

```text
attempt_id
inference_id
purpose = "inference" | "summary"
provider
model_id
model
wire_version
raw_estimate
factor
payload_bytes
payload_sha256
```

В terminal-записи сохранять nullable usage и классификацию ошибки.

Расширить `LLMResponse` так, чтобы неизвестные `prompt_tokens` и `completion_tokens` оставались `None`. Сейчас неизвестный usage заменяется нулём: `src/agent/llm.py:209`, `src/agent/sse.py:188`.

Для summary использовать тот же audit. Не сохранять исходный prompt или текст summary в `detail`.

Использовать существующий `OutboundAudit`: `src/agent/models.py:168`. Полный финансовый ledger не добавлять в `0008`.

Закрывать read-транзакцию перед HTTP. Контекстный менеджер DB session сам по себе не означает отсутствие транзакции после SELECT. ([docs.sqlalchemy.org](https://docs.sqlalchemy.org/en/20/orm/session_basics.html))

---

## 6. Эндпоинт `/context`

Сохранить маршрут:

```text
GET /api/sessions/{session_id}/context
```

Использовать тот же loader, построитель wire-запроса, reasoning policy, tools, budget и calibrator.

Вызвать:

```python
await manager.prepare(
    history,
    model,
    definitions,
    effective_max_tokens=model.max_tokens,
    reasoning_effort=decision.effective,
    summary_runner=None,
    stream=True,
)
```

Эндпоинт:

- Читает существующий checkpoint.
- Применяет ступени 1–2.
- Применяет детерминированную ступень.
- Не вызывает модель.
- Не сохраняет checkpoint.
- Не обновляет калибровку.
- Не создаёт audit и события.

Сохранить существующие поля ответа:

```text
approximate_tokens
omitted_messages
truncated_tool_results
context_window
reserved_tokens
```

Сохранить смысл `reserved_tokens` как `context_window - input_tokens`. Он включает output reservation и safety margin.

Добавить:

```text
model_id
provider
wire_version
raw_tokens
tokens_before
calibration_factor
calibration_samples
input_budget
output_tokens
effective_max_tokens
safety_margin
target_tokens
stage
fits
masked_tool_results
folded_tool_calls
summary_used
summary_calls
summary_checkpoint_id
covers_until
would_summarize
error
```

`summary_calls` всегда равен нулю.

При `ContextBudgetError` вернуть `200` с `fits=false` и диагностикой из `error.result`. Не превращать ожидаемую невозможность уложить контекст в `500`.

Для отсутствующей сессии сохранить `404`.

Preview и runtime используют одинаковую оценку. Результат будущей платной суммаризации может изменить runtime-проекцию. Отражать это через `would_summarize=true`.

---

## 7. События и UI

### 7.1. `context.compacted`

Сохранить тип события и три существующих поля: `src/agent/runtime.py:389`.

Добавить:

```json
{
  "inference_id": "...",
  "model_id": "...",
  "provider": "...",
  "wire_version": "...",
  "reason": "budget",
  "stage": "summary",
  "tokens_before": 18000,
  "raw_tokens": 9000,
  "approximate_tokens": 10800,
  "calibration_factor": 1.2,
  "calibration_samples": 5,
  "input_budget": 20000,
  "target_tokens": 15000,
  "effective_max_tokens": 4000,
  "safety_margin": 640,
  "omitted_messages": 24,
  "truncated_tool_results": 3,
  "masked_tool_results": 3,
  "folded_tool_calls": 1,
  "summary_used": true,
  "summary_calls": 1,
  "summary_checkpoint_id": 101,
  "covers_until": 80,
  "overflow_retry": 1
}
```

Допустимые `reason`:

```text
budget
overflow
output_budget
fallback
checkpoint
```

Публиковать одно событие для итоговой изменённой проекции. Не публиковать событие каждой пробной ступени.

Перед событием сохранить новый checkpoint. Не публиковать незафиксированный ID.

### 7.2. Дополнительные события

Добавить:

- `context.overflow` — обнаружен typed overflow; передать model, code, target и номер следующего повтора.
- `context.error` — контекст не помещается; передать reason и бюджет.
- `context.summary_failed` — платная ступень не дала пригодный результат; передать kind ошибки и число вызовов.

Не включать error body, исходные сообщения и summary-текст в payload.

Сохранить существующий `session.error` для terminal-состояния.

### 7.3. Минимальные изменения UI

Расширить `ContextStats`: `frontend/src/api.ts:33`.

Добавить `kind` и метаданные summary в тип `Message`: `frontend/src/api.ts:59`, `src/agent/schemas.py:56`.

Показывать checkpoint отдельной карточкой. Не показывать его как новый запрос пользователя.

Показывать:

- Оценку относительно input budget.
- Применённую ступень.
- Число заменённых сообщений.
- Использование checkpoint.
- Номер overflow-повтора.

Обрабатывать `context.compacted` адресно. Не запускать полный `refresh()` для каждого события контекста. Текущий обработчик обновляет всё состояние: `frontend/src/App.vue:169`.

---

## 8. Тесты и приёмка

Имена ниже задают новые проверки интеграции. Существующие тесты чистых модулей сохраняют собственную область ответственности.

### 8.1. Unit

| Имя теста | Проверяемое поведение |
|---|---|
| `test_prepare_keeps_uncompressed_context_when_it_fits` | Большой tool-результат остаётся неизменным, если весь wire помещается |
| `test_prepare_estimates_tools_and_reasoning` | Оценка включает tools, arguments, reasoning и overhead |
| `test_prepare_adjusts_complete_wire_once` | `adjust()` применяется один раз к полной raw-оценке |
| `test_prepare_uses_effective_output_budget` | Повышение `max_tokens` уменьшает input budget |
| `test_prepare_never_clamps_invalid_budget_upward` | Невозможный бюджет не превращается в `1024` |
| `test_prepare_preserves_all_protected_blocks` | Защищённые данные не изменяются на любой ступени |
| `test_prepare_keeps_large_exchange_atomic` | Exchange с несколькими tool-ответами не разрывается |
| `test_prepare_stage_order_stops_at_first_fit` | Следующие ступени не запускаются после успешной |
| `test_prepare_stage_two_does_not_remask_results` | Маскирование не применяется повторно |
| `test_prepare_rejects_expanding_compaction` | Преобразование, увеличившее wire, отбрасывается |
| `test_prepare_checks_protected_minimum_before_summary` | Невозможный защищённый минимум не вызывает модель |
| `test_summary_renders_data_before_current_user` | Summary получает `role="user"` и маркер данных |
| `test_summary_does_not_become_reasoning_override` | Summary не меняет последний настоящий user |
| `test_summary_coverage_uses_message_ids` | Coverage использует DB ID, а не позиции блоков |
| `test_summary_hash_uses_original_prefix` | Хеш не зависит от маскирования и модели inference |
| `test_summary_chunks_use_light_budget_without_recursion` | Чанки состоят из целых блоков; рекурсии нет |
| `test_summary_failure_uses_deterministic_fold` | Ошибка summary запускает бесплатную ступень |
| `test_summary_cancellation_propagates` | Отмена не превращается в fallback или новый summary |
| `test_overflow_target_forces_smaller_projection` | Повтор получает меньший target и raw-размер |
| `test_overflow_classifier_rejects_generic_400` | Обычный `400` не запускает сжатие |
| `test_stream_tool_delta_marks_request_started` | Tool-дельта запрещает любой автоматический повтор |
| `test_calibration_key_separates_provider_and_wire_version` | Ключи не смешивают разные destinations и версии |
| `test_calibration_ignores_unknown_or_nonpositive_usage` | Samples и factor не меняются |
| `test_routing_light_reference_and_null_semantics` | Неизвестная ссылка отклоняется; null выбирает default |
| `test_registry_roundtrip_preserves_light` | Все конструкторы и обратные преобразования сохраняют поле |

Заменить тест строковой обрезки tool-результата: `tests/unit/test_context.py:22`.

Сохранить проверки чистых функций маскирования и EWMA: `tests/unit/test_context_compaction.py:25`, `tests/unit/test_token_calibration.py:16`.

### 8.2. Integration

| Имя теста | Проверяемое поведение |
|---|---|
| `test_migration_0008_upgrade_preserves_original_messages` | Upgrade сохраняет данные и добавляет поля, индексы и constraints |
| `test_migration_0008_downgrade_removes_only_checkpoints` | Downgrade удаляет производные summary, но сохраняет исходники |
| `test_migration_0008_metadata_matches_models` | Миграция соответствует ORM на SQLite и PostgreSQL |
| `test_summary_checkpoint_is_saved_before_inference` | Inference использует checkpoint с durable ID |
| `test_checkpoint_reuse_sends_cover_and_tail_without_duplicates` | Старый prefix заменён; anchors и хвост не дублируются |
| `test_checkpoint_restart_loads_protected_and_tail_pages` | Cold start не материализует весь архив без необходимости |
| `test_duplicate_checkpoint_insert_reuses_existing_row` | Конкурентная вставка не создаёт дубликат |
| `test_summary_limit_is_shared_by_primary_overflow_and_fallback` | Общий лимит не сбрасывается между путями |
| `test_summary_attempts_are_audited_individually` | Каждый HTTP-вызов имеет собственные status, hash и usage |
| `test_audit_hash_matches_transmitted_body` | Audit хеширует фактически отправленные байты |
| `test_calibration_updates_once_per_http_attempt` | Повторный terminal callback не увеличивает samples |
| `test_concurrent_calibration_updates_do_not_lose_samples` | CAS сохраняет оба наблюдения |
| `test_calibration_survives_restart` | Новый процесс использует сохранённый factor |
| `test_overflow_allows_initial_request_and_two_retries` | Четвёртая overflow-отправка не возникает |
| `test_overflow_stops_on_same_payload` | Повторный fingerprint завершает цикл |
| `test_generic_400_does_not_summary_retry_or_fallback` | Validation error остаётся terminal |
| `test_fallback_reprepares_context_for_actual_model` | Fallback использует собственный budget, factor и reasoning |
| `test_fallback_persists_actual_reasoning_decision` | Сообщение получает параметры фактической модели |
| `test_stream_failure_after_delta_never_retries_or_falls_back` | Частичный поток не запускает новую генерацию |
| `test_stream_length_after_delta_does_not_raise_output_budget` | Обрезанный поток не повторяется автоматически |
| `test_stop_during_summary_prevents_checkpoint_and_inference` | Stop отменяет продолжение подготовки |
| `test_context_endpoint_has_no_network_or_write_side_effects` | GET не вызывает HTTP, не пишет summary, audit или calibration |
| `test_context_preview_matches_runtime_without_new_summary` | При одинаковом snapshot бесплатные пути дают одинаковую оценку |
| `test_context_endpoint_reports_nonfitting_protected_data` | Возвращаются `fits=false` и диагностика, а не `500` |
| `test_context_events_preserve_legacy_fields` | Новые payload совместимы с текущими полями UI |

Для HTTP использовать заглушки. Платные вызовы не нужны для обязательной приёмки.

### 8.3. Критерии готовности

- Primary, fallback и `/context` используют один оценщик.
- Tools входят в бюджет.
- Защищённые блоки сохраняются.
- Summary и детерминированная свёртка сохраняют coverage.
- Калибровка переживает рестарт.
- Каждый HTTP attempt попадает в audit.
- Overflow допускает не более двух повторов.
- Первая содержательная дельта закрывает все пути повтора.
- Upgrade и downgrade проходят на обеих БД.
- Исходные сообщения не меняются ради сжатия.

---

## 9. Риски и решения для подтверждения

| Решение | Рекомендуемый вариант | Риск |
|---|---|---|
| `light` не задан | Использовать `routing.default` | Суммаризация может оказаться дорогой |
| Область лимита summary | Четыре вызова на логический inference | Длинная сессия может выполнить несколько таких групп |
| Overflow и fallback | Не переключать модель из-за overflow | Модель с большим окном не спасает запрос автоматически |
| Строковая обрезка tools | Удалить из подготовки контекста | Старый тест и смысл `truncated_tool_results` меняются |
| Повтор после streaming | Запретить после первой содержательной дельты | Обрезанный ответ потребует отдельного продолжения |
| Проверка source hash | Append-only контракт; инвалидировать при будущих edit/skip | Внешняя правка БД не обнаруживается без чтения prefix |
| Детерминированный checkpoint | Сохранять с `model=NULL` | Сжатие остаётся потерей части подробностей |
| Защита от конкурентного Stop | Добавить проверки snapshot и условные записи сейчас | Полная защита от старого владельца требует generation/lease |

Эти варианты задают рабочий контракт реализации. Изменение любого варианта требует обновить соответствующие тесты до начала интеграции.
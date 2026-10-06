# Фаза 2 — состояние работ

Дата: 2026-10-06. Ветка: `dev`. План шагов — в разделе 9 отчёта, детали — в `phase-2-spec.md`.

## Сделано (не закоммичено)

- **2.1 Рассуждения.**
  - Миграция 0007: `messages.kind`, `reasoning_content`, `reasoning_effort`.
  - Модуль `reasoning.py`: выбор уровня (сообщение → сессия → `auto` или модель) и адаптеры
    `reasoning_effort`, `enable_thinking`, `thinking.budget_tokens`.
  - Политика истории `reasoning.history`: `none`, `current_turn`, `all`.
  - Уровень на сообщение: `POST /api/sessions/{id}/messages` с полем `reasoning_effort`.
  - Проверка на шлюзе LiteLLM: `reasoning_effort: "none"` выключает рассуждения, другие уровни
    только включают их. Поэтому у deepseek уровни `none` и `high`.
- **2.2 Стриминг.**
  - `sse.py`: декодер SSE и сборщик чанков. Разрыв `\r\n` между чанками обрабатывается.
  - `LLMClient.chat(on_delta=...)` стримит ответ. Повтор возможен только до первого фрагмента.
  - `StreamPublisher`: временные события `message.delta`, `reasoning.delta`, `stream.reset`
    каждые 50 мс. Строк в таблице `events` эти события не создают.
  - UI показывает набираемый ответ.
- **B8:** shell получает `HOME`, `USER`, `LOGNAME`.
- Проверки: 340 тестов, `ruff`, `mypy`, сборка фронтенда, живой тест стриминга.

## Готовые модули (ещё не подключены)

Каждый модуль покрыт тестами и проходит `mypy --strict`.

- `context_compaction.py` — ступени 1–2 сжатия: структурное маскирование старых результатов
  инструментов и свёртка больших аргументов `write_file` / `edit_file`.
- `token_calibration.py` — калибровка оценки токенов по `usage.prompt_tokens` (EWMA,
  диапазон 0.5–4.0).
- `usage.py` — usage из ответа и стоимость через `Decimal`; «неизвестно» не равно нулю.
- `retry_policy.py` — `Retry-After`, backoff с jitter, circuit breaker на провайдера.

## Следующие шаги

1. 2.3: подключить `context_compaction` и `token_calibration` в `ContextManager`.
   Суммаризация моделью `light`. Распознавание переполнения контекста и не более двух повторов.
2. 2.4: дерево сообщений (`parent_id`, ветка) и эндпоинты edit, regenerate, fork.
3. 2.5: дочерние агенты без вложенного цикла, статус `waiting_child`.
4. 2.6: таблица попыток с `usage.py`; `retry_policy` в `LLMClient`; параллельные read-only
   инструменты.
5. 2.7: надёжность Redis — generation, `wake_seq`, outbox, курсор `XAUTOCLAIM`, наблюдатель
   отмены. Объём большой; делать отдельными маленькими шагами.
6. Сохранять частичный ответ оборванного стрима как сообщение `kind="interrupted"`.

## Распределение работы

- Изолированные модули с тестами: отдельный worktree на модуль. Запись только в указанные
  файлы, shell только для pytest, ruff, mypy.
- Спецификации и ревью: отдельный этап перед интеграцией.
- Интеграция в `runtime.py`, `api.py`, миграции: основная ветка `dev`.

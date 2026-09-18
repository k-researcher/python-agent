# Python Agent

Самостоятельный локальный ИИ-агент на Python.

## Что реализовано

- FastAPI REST API и WebSocket на одном порту;
- SQLite по умолчанию либо PostgreSQL для состояния runtime;
- проекты, родительские и дочерние сессии, сообщения, tool calls, approvals и события;
- восстановление запущенных сессий в статус `interrupted` после перезапуска;
- OpenAI-совместимый `/chat/completions`, retry, таймауты и ограничение контекста;
- режимы `dev` и `ask`, централизованные подтверждения до побочного эффекта;
- безопасные файловые инструменты и запуск команд внутри корня проекта;
- HTTP, web search, именованные подключения к БД, SSH и чтение XLSX;
- опциональная база знаний с локальным лексическим поиском или удалёнными embeddings;
- локальный журнал исходящих обращений без сохранения передаваемого текста;
- Vue 3 + TypeScript UI с approvals, дочерними агентами, context stats и audit;
- API token обязателен при bind на внешний интерфейс;
- опциональный distributed runtime: Redis Streams, PostgreSQL и отдельные workers.
- одновременные именованные LLM-профили на уровне сессии и дочернего агента.

## Быстрый запуск

Требуются Python 3.12+, `uv` и Node.js 20+.

```bash
cd /Users/oxygen/Files/Code/ml/agents/python-agent
uv sync --extra dev
cp .env.example .env
cd frontend && npm install && npm run build && cd ..
uv run agent
```

Перед запуском заполните как минимум:

```dotenv
AGENT_LLM_BASE_URL=https://your-provider.example/v1
AGENT_LLM_API_KEY=replace-me
AGENT_LLM_MODEL=your-model
```

### Несколько моделей одновременно

Одиночные `AGENT_LLM_*` остаются обратносуместимым профилем `default`. Чтобы разные
сессии и дочерние агенты одновременно использовали разные модели или провайдеров,
задайте JSON-профили:

```dotenv
AGENT_DEFAULT_LLM_PROFILE=fast
AGENT_LLM_PROFILES={"fast":{"base_url":"https://provider-a.example/v1","api_key":"key-a","model":"fast-model"},"reasoning":{"base_url":"https://provider-b.example/v1","api_key":"key-b","model":"reasoning-model","max_tokens":16000},"local":{"base_url":"http://host.docker.internal:11434/v1","api_key":"local","model":"local-model"}}
```

Профиль выбирается при создании сессии. `run_agent` наследует профиль родителя либо
получает другой `llm_profile`. Redis хранит только session ID; worker читает имя
профиля из PostgreSQL. Названия, модели и hostname видны в public config, API keys —
нет. После изменения профилей API и workers необходимо перезапустить.

Откройте `http://127.0.0.1:8080`. OpenAPI доступен на
`http://127.0.0.1:8080/docs`. Миграции применяются автоматически.

## Где находятся данные

По умолчанию всё состояние находится в `data/agent.db`. Диалоги не лежат в workers
или в файловом session cache: они записываются в таблицы `sessions` и `messages`.
Файлы проекта читаются только по запросу инструмента и не копируются в отдельное
хранилище автоматически.

Во внешнюю сеть всегда уходит активный контекст при вызове настроенного LLM — иначе
удалённая модель не сможет ответить. Остальные сетевые инструменты выключены по
умолчанию и скрыты от модели. База знаний также выключена. Никакой автоматической
отправки диалогов в KB или на отдельный knowledge-сервер нет.

Таблица `outbound_audit` хранит направление, операцию, число байт, SHA-256, статус и
ошибку. Сам prompt, тело HTTP-запроса и содержимое документа в audit не сохраняются.

## Политика инструментов

- read-only операции выполняются без подтверждения;
- запись файлов, shell, HTTP, SSH, изменение БД, удаление и запуск ребёнка требуют
  явного Approval в UI;
- `ask` не получает инструменты изменения локального проекта;
- отключённые network/KB capabilities вообще не передаются модели;
- HTTP разрешён только для `AGENT_NETWORK_ALLOWLIST`, проверяет DNS и блокирует
  private/loopback/link-local адреса, redirect не выполняется;
- пути файлов и XLSX ограничены зарегистрированным корнем проекта с проверкой symlink;
- `db_select` принудительно включает read-only transaction для SQLite/PostgreSQL;
- SSH требует заранее настроенный `known_hosts`.

## Сеть, БД и SSH

Значения-коллекции задаются JSON:

```dotenv
AGENT_ALLOW_NETWORK_TOOLS=true
AGENT_NETWORK_ALLOWLIST=["api.example.com","*.search.example"]
AGENT_WEB_SEARCH_URL_TEMPLATE=https://search.example/api?q={query}
AGENT_DATABASE_CONNECTIONS={"reporting":"postgresql+asyncpg://user:pass@db/reporting"}
AGENT_SSH_CONNECTIONS={"stage":{"host":"stage.example.com","username":"deploy","known_hosts":"/absolute/path/known_hosts","client_keys":["/absolute/path/id_ed25519"]}}
```

Модель видит только имена подключений (`reporting`, `stage`), но не URL, пароль и
пути ключей. Для HTTP allowlist обязателен; именованные DB/SSH endpoints являются
отдельным явным allowlist администратора.

## База знаний

Локальный вариант не делает сетевых запросов:

```dotenv
AGENT_KNOWLEDGE_BASE_ENABLED=true
```

Для семантического поиска можно дополнительно настроить OpenAI-совместимый embedding
endpoint и добавить его hostname в network allowlist:

```dotenv
AGENT_ALLOW_NETWORK_TOOLS=true
AGENT_NETWORK_ALLOWLIST=["embeddings.example.com"]
AGENT_KNOWLEDGE_BASE_EMBEDDING_URL=https://embeddings.example.com/v1/embeddings
AGENT_KNOWLEDGE_BASE_EMBEDDING_API_KEY=replace-me
AGENT_KNOWLEDGE_BASE_EMBEDDING_MODEL=embedding-model
```

Поиск с удалёнными embeddings и любые изменения KB проходят через Approval. Агент не
индексирует диалоги самостоятельно.

## Внешний доступ

При `AGENT_HOST=0.0.0.0` или другом нелокальном bind запуск без
`AGENT_API_TOKEN` завершится ошибкой. REST использует заголовок `X-Agent-Token`.
WebSocket передаёт token первым JSON-сообщением после соединения, поэтому секрет не
попадает в URL/access-log. Для внешней сети следует ставить TLS reverse proxy.

## Локальный и распределённый runtime

По умолчанию используется `AGENT_EXECUTION_MODE=embedded`: API и выполнение агента
находятся в одном процессе, SQLite разрешён, Redis не нужен. Это оптимальный режим
для одного пользователя.

Для нескольких процессов используется `AGENT_EXECUTION_MODE=redis`. В этом режиме:

- API записывает задания в Redis Stream и не исполняет tool calls;
- несколько `agent-worker` читают одну consumer group;
- PostgreSQL является общим источником состояния;
- Redis Pub/Sub доставляет события от workers всем WebSocket-инстансам API;
- heartbeat позволяет readiness endpoint видеть активные workers;
- stale pending jobs забираются другим worker после visibility timeout;
- распределённая блокировка не даёт двум workers вести одну сессию одновременно.

Redis-режим намеренно откажется запускаться с SQLite.

### Запуск через Docker Compose

Укажите в `.env` LLM-настройки, непустой `AGENT_API_TOKEN` и желательно смените пароль:

```dotenv
AGENT_API_TOKEN=replace-with-long-random-token
AGENT_POSTGRES_PASSWORD=replace-me
AGENT_REDIS_PASSWORD=replace-with-random-hex
AGENT_PROJECTS_ROOT=/absolute/path/with/projects
```

Compose по умолчанию публикует API только на `127.0.0.1`, запускает agent-контейнеры
без root, без Linux capabilities, с read-only root filesystem и защищает Redis
паролем. Для reverse proxy можно явно установить `AGENT_BIND_ADDRESS=0.0.0.0`.

Затем:

```bash
make docker-up WORKERS=2
make docker-status
```

`make docker-up` выполняет `docker compose up`, дожидается успешного migration job и
затем удаляет остановленный контейнер `migrate` командой
`docker compose rm -f migrate`. PostgreSQL и применённая схема не удаляются.

Стандартный диагностический запуск также доступен:

```bash
docker compose up --build -d
```

В этом случае `migrate` останется видимым как `Exited (0)`, что означает успешное
применение миграций.

Добавить workers:

```bash
make docker-up WORKERS=3
```

Все workers должны видеть проекты по одному и тому же пути. Compose монтирует
`AGENT_PROJECTS_ROOT` как `/workspace`, поэтому в UI следует регистрировать пути вида
`/workspace/project-name`.

Остановка распределённой сессии кооперативная: worker увидит её между операциями.
Уже начавшийся HTTP/LLM вызов может закончиться до установленного timeout. Если worker
погиб во время побочного tool call, runtime помечает вызов ошибкой и не повторяет его
автоматически — это защищает от двойной записи или повторной команды SSH.

## Проверки

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
uv run alembic check
cd frontend && npm run typecheck && npm run build
```

## Ресурсы

Embedded-режим почти не изменился по накладным расходам. Distributed-режим добавляет
PostgreSQL, Redis и минимум один worker. Для небольшой установки разумно заложить
примерно 0.5–1.5 ГБ RAM суммарно без учёта локальной LLM; фактический расход зависит
от числа workers, параллельных сессий и размера Excel/tool outputs.

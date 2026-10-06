# Python Agent

Самостоятельный локальный ИИ-агент на Python.

## Что реализовано

- FastAPI REST API и WebSocket на одном порту.
- SQLite по умолчанию или PostgreSQL для состояния runtime.
- Проекты, родительские и дочерние сессии, сообщения, вызовы инструментов, подтверждения и
  события.
- Перевод запущенных сессий в статус `interrupted` после перезапуска.
- Модели и провайдеры в `config/models.yaml` плюс переопределения из UI. У каждой модели своё
  окно контекста и свои уровни рассуждений. Есть запасная модель на случай сбоя основной.
- OpenAI-совместимый `/chat/completions`: повторы, таймауты, повтор обрезанного ответа с
  большим лимитом токенов.
- Режимы `dev` и `ask`. Подтверждение до каждого побочного эффекта. Проверка аргументов по
  JSON Schema до подтверждения.
- Безопасные файловые инструменты (`SafeProjectFS`) и запуск команд внутри корня проекта.
- Stop останавливает агента: проверка перед каждым инструментом, отмена ожидающих
  подтверждений, завершение группы процессов shell.
- HTTP, web search, именованные подключения к БД, SSH и чтение XLSX.
- Необязательная база знаний с локальным поиском по ключевым словам или с внешними
  эмбеддингами.
- Локальный журнал исходящих обращений без сохранения передаваемого текста.
- Vue 3 + TypeScript UI: подтверждения, дочерние агенты, статистика контекста, журнал.
- Вход по одноразовой ссылке и cookie-сессия; токен в UI не нужен.
- Необязательный распределённый runtime: Redis Streams, PostgreSQL и отдельные workers.

Архитектурные решения: [docs/adr](docs/adr). Безопасность: [docs/security.md](docs/security.md),
[docs/threat-model.md](docs/threat-model.md).

## Быстрый запуск

Требуются Python 3.12+, `uv` и Node.js 20+.

```bash
cd python-agent
uv sync --extra dev
cp .env.example .env
cd frontend && npm install && npm run build && cd ..
uv run agent
```

Перед запуском заполните ключ провайдера в `.env` и опишите модели в
`config/models.yaml` (шаблон — `config/models.example.yaml`):

```bash
cp config/models.example.yaml config/models.yaml
echo 'AGENT_LLM_API_KEY=...' >> .env
uv run agent config check --strict
```

### Модели и провайдеры

Конфигурация моделей состоит из двух слоёв:

1. **`config/models.yaml`** — провайдеры, модели, маршрутизация. Ключей в файле нет: провайдер
   ссылается на переменную окружения через `api_key_env`.
2. **Переопределения из UI** (`/api/settings/models`) — хранятся в БД, общие для API и всех
   workers. Перекрывают YAML по отдельным полям, могут отключать элементы файла, добавлять
   новых провайдеров и модели и хранить ключи зашифрованными (мастер-ключ
   `AGENT_SECRET_KEY`; в embedded-режиме при его отсутствии создаётся `data/secret.key`).
   «Сбросить к файлу» удаляет переопределение.

Типы провайдеров: `openai` — любой OpenAI-совместимый `/chat/completions` (корпоративные шлюзы, GLM,
LM Studio, `mlx_lm.server`, Ollama); `openai_responses` (Codex/GPT) и `anthropic` (Claude)
объявляются заранее, их адаптеры появятся в следующей фазе. У каждой модели свои окно
контекста, `max_tokens` и допустимые уровни рассуждений; уровень выбирается при создании
сессии (`reasoning_effort`). Модель сессии: явная → по роли (`routing.roles`) → `routing.default`.

Если `config/models.yaml` нет, используются прежние `AGENT_LLM_*`; JSON-строка
`AGENT_LLM_PROFILES` устарела.

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

## Вход и внешний доступ

Токен в интерфейсе вводить не нужно. При запуске, если активного входа ещё нет, агент
печатает одноразовую ссылку вида `http://127.0.0.1:8080/#login=...` (на localhost она
открывается в браузере автоматически). Код передаётся во fragment и не попадает в логи;
после входа браузер получает `HttpOnly; SameSite=Strict` cookie, а мутирующие запросы
дополнительно защищены CSRF-токеном и проверкой `Origin`.

```bash
uv run agent auth link        # новая ссылка (например, для другого устройства)
uv run agent auth revoke-all  # выйти на всех устройствах
```

Все запросы проверяют точный `Host` (`AGENT_ALLOWED_HOSTS`) — это защищает от DNS
rebinding. Для удалённого доступа поставьте TLS reverse proxy и задайте:

```dotenv
AGENT_PUBLIC_ORIGIN=https://agent.example.com
AGENT_ALLOWED_HOSTS=["agent.example.com"]
AGENT_TRUSTED_PROXY_IPS=["127.0.0.1"]
```

При HTTPS cookie получает префикс `__Host-` и флаг `Secure`. `AGENT_API_TOKEN` остаётся
необязательным способом аутентификации для CLI и скриптов (`X-Agent-Token`). Для
разработки фронтенда через Vite добавьте `AGENT_EXTRA_ORIGINS=["http://localhost:5173"]`.

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

Укажите в `.env` LLM-настройки и желательно смените пароль:

```dotenv
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

Тесты не читают `.env` и не трогают `data/agent.db`: `tests/conftest.py` задаёт
`AGENT_ENV_FILE=` (пустое значение отключает dotenv) и временную SQLite-базу.

## Ресурсы

Embedded-режим почти не изменился по накладным расходам. Distributed-режим добавляет
PostgreSQL, Redis и минимум один worker. Для небольшой установки разумно заложить
примерно 0.5–1.5 ГБ RAM суммарно без учёта локальной LLM; фактический расход зависит
от числа workers, параллельных сессий и размера Excel/tool outputs.

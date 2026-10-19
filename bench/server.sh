#!/usr/bin/env bash
# Отдельный сервер агента для прогона эталонных задач bench.
#
# Использует свой датакаталог data/bench/server, свою SQLite-базу (миграции
# применяются при старте), обязательную песочницу shell и не открывает браузер.
# Порт фиксирован: 8098. Лог пишется в data/bench/server/server.log; скрипт
# печатает одноразовую ссылку входа (#login=...) из этого лога.
#
# Запуск от владельца репозитория вручную:
#   bash bench/server.sh
# После старта в консоли появится ссылка входа, которую прочитает bench/run.py.

set -euo pipefail

# Корень репозитория (родитель каталога bench/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AGENT_HOST="${AGENT_HOST:-127.0.0.1}"
export AGENT_PORT="${AGENT_PORT:-8098}"

# Своя SQLite-база и каталог данных для bench.
export AGENT_DATA_DIR="${ROOT}/data/bench/server"
export AGENT_DATABASE_URL="sqlite+aiosqlite:///${ROOT}/data/bench/server/bench.db"

# Shell всегда в песочнице ОС; читаемых путей вне проекта не добавляем.
export AGENT_SHELL_SANDBOX="required"
export AGENT_SHELL_SANDBOX_READ_PATHS="[]"
export AGENT_AUTH_OPEN_BROWSER="false"

# Модели берутся из обычной конфигурации репозитория (config/models.yaml или AGENT_LLM_*),
# поэтому ключи и профили здесь не задаются.

mkdir -p "${AGENT_DATA_DIR}"
LOG_FILE="${AGENT_DATA_DIR}/server.log"
: > "${LOG_FILE}"

cd "${ROOT}"

echo "bench server: запуск на http://127.0.0.1:${AGENT_PORT} (лог: ${LOG_FILE})"

"${ROOT}/.venv/bin/pyagent" >> "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

# Ожидаем строку входа в логе (сервер печатает её при первом старте).
for _ in $(seq 1 60); do
    if grep -q "#login=" "${LOG_FILE}"; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "bench server: процесс завершился досрочно, лог:" >&2
        tail -n 40 "${LOG_FILE}" >&2
        exit 1
    fi
    sleep 1
done

if grep -q "#login=" "${LOG_FILE}"; then
    echo "bench server: одноразовая ссылка входа:"
    grep "#login=" "${LOG_FILE}" | tail -n 1
fi

wait "${SERVER_PID}"

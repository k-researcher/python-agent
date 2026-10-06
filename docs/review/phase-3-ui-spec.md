# Фаза 3 — новый UI: шаги 3.1–3.2

Дата: 2026-10-06. Основа — раздел 6 отчёта `2026-10-06-architecture-review.md`. Целевой дизайн —
светлый макет (локальный файл, вне репозитория). Язык интерфейса — русский.

## 1. Стек и правила

- Vue 3.5 + TypeScript (strict), `<script setup lang="ts">`, Vite.
- `vue-router` 4, `pinia` 3, `@vueuse/core`, иконки `lucide-vue-next`.
- Markdown: `markdown-it` с `html: false`, результат проходит `DOMPurify`. Подсветка кода —
  `shiki` (ленивая загрузка, языки по запросу). Сырой HTML от модели не вставляется никогда.
- Тесты: `vitest` + `@vue/test-utils` + `happy-dom`, файлы `src/**/*.test.ts`.
- Проверка: `npm run build` (включает `vue-tsc`), `npm test`.
- Нет внешних шрифтов и CDN: всё работает офлайн.

## 2. Структура каталогов

```text
src/
  main.ts               создание app, pinia, router
  router.ts             маршруты
  api/types.ts          типы ответов API
  api/client.ts         request(), api.* (перенос из src/api.ts)
  realtime.ts           WebSocket с переподключением
  stores/app.ts         конфигурация, проекты, сессии, статус соединения
  stores/session.ts     открытая сессия: сообщения, подтверждения, стриминг
  styles/tokens.css     дизайн-токены (светлая и тёмная тема)
  styles/base.css       сброс, типографика, утилиты
  components/layout/    AppShell, Topbar, LeftSidebar, RightPanel
  components/chat/      ChatTimeline, MessageItem, ReasoningBlock, ToolCallCard,
                        ApprovalInline, Composer, EmptyState, MarkdownView
  views/                SessionView, HomeView, LoginView
```

`src/auth.ts` остаётся и используется `LoginView`.

## 3. Маршруты

| Путь | Вид |
|---|---|
| `/` | `HomeView`: пустое состояние «Что будем делать?» и composer новой сессии |
| `/s/:id` | `SessionView`: лента сессии и composer |
| `/login` | `LoginView`: вход по ссылке; без ссылки — подсказка `agent auth link` |

Без входа (401) любой маршрут ведёт на `/login`.

## 4. Контракт хранилищ

```ts
// stores/app.ts
export const useAppStore = defineStore("app", () => {
  const config: Ref<PublicConfig | null>;
  const projects: Ref<Project[]>;
  const sessions: Ref<Session[]>;
  const connection: Ref<"connecting" | "live" | "offline">;
  const selectedProjectId: Ref<string | null>;
  async function load(): Promise<void>;                  // config + projects + sessions
  async function createProject(name: string, rootPath: string): Promise<Project>;
  async function createSession(input: NewSession): Promise<Session>;
  function sessionTree(projectId: string | null): SessionNode[];  // parent → children
  function applyEvent(event: ServerEvent): void;        // session.*, project.*
  return { ... };
});

// stores/session.ts
export const useSessionStore = defineStore("session", () => {
  const sessionId: Ref<string | null>;
  const messages: Ref<Message[]>;
  const approvals: Ref<Approval[]>;
  const streaming: Ref<{ content: string; reasoning: string } | null>;
  const context: Ref<ContextStats | null>;
  const audit: Ref<OutboundAudit[]>;
  const children: Ref<Session[]>;
  async function open(id: string): Promise<void>;
  async function send(content: string, reasoningEffort?: string): Promise<void>;
  async function resolve(approvalId: string, decision: "approve" | "reject", comment?: string): Promise<void>;
  async function stop(): Promise<void>;
  function applyEvent(event: ServerEvent): void;        // message.*, reasoning.delta, stream.reset,
                                                         // approval.*, tool.completed, context.compacted
  const timeline: ComputedRef<TimelineItem[]>;          // см. раздел 5
  return { ... };
});
```

`TimelineItem` — объединённая лента:

```ts
type TimelineItem =
  | { kind: "user"; message: Message }
  | { kind: "assistant"; message: Message }                 // текст и reasoning
  | { kind: "tool"; call: ToolCallView }                    // вызов + результат + подтверждение
  | { kind: "summary"; message: Message }                   // checkpoint контекста
  | { kind: "streaming"; content: string; reasoning: string };

interface ToolCallView {
  id: string; name: string; arguments: Record<string, unknown>; argumentsRaw: string;
  result: { success: boolean; output?: unknown; error?: string } | null;  // из tool-сообщения
  approval: Approval | null;                                 // ожидающее подтверждение
  status: "pending" | "awaiting_approval" | "running" | "completed" | "error" | "rejected";
}
```

Правила `timeline`: сообщения `skipped` не показываются; assistant с `tool_calls` даёт свой текст
(если есть) и по одному элементу `tool` на вызов; результат берётся из сообщения `role="tool"`
с тем же `tool_call_id`; подтверждение — из `approvals` по `tool_call_id`.

## 5. События WebSocket

Сервер шлёт JSON на `/ws`. Постоянное событие:
`{ "id": number, "type": string, "session_id": string | null, "payload": object, "created_at": string }`.
Временное событие стриминга (без строки в БД):
`{ "type": "message.delta" | "reasoning.delta" | "stream.reset", "session_id": string,
"ephemeral": true, "payload": { "stream_id": string, "seq"?: number, "delta"?: string } }`.
Дельта с `stream_id`, отличным от текущего, начинает новый поток; `seq` растёт внутри потока,
повтор или старый `seq` пропускается.

| Событие | Действие клиента |
|---|---|
| `message.delta`, `reasoning.delta` | дописать текст в `streaming` (только для открытой сессии) |
| `stream.reset` | очистить `streaming` |
| `message.created` | перечитать сообщения открытой сессии; очистить `streaming` |
| `approval.required`, `approval.resolved`, `approval.cancelled` | перечитать подтверждения |
| `tool.completed` | перечитать сообщения |
| `session.status`, `session.error`, `session.created`, `session.archived`, `session.child_created` | обновить список сессий; для открытой — статус |
| `context.compacted` | перечитать `context` |
| `project.created` | перечитать проекты |

Перечитывание адресное: не весь `load()`. Переподключение: 1 с, 2 с, 4 с … максимум 30 с; после
переподключения — `load()` и `open()` текущей сессии.

## 6. Раскладка и токены

- Сетка: `Topbar` 52px; левая колонка 264px; центр — гибкий; правая колонка 320px.
  Ниже 1100px правая колонка скрывается за кнопкой; ниже 760px левая — тоже.
- Светлая тема по умолчанию, тёмная по `prefers-color-scheme` и переключателю.
- Токены (CSS-переменные): `--bg`, `--bg-subtle`, `--surface`, `--border`, `--text`,
  `--text-muted`, `--accent`, `--success`, `--warning`, `--danger`, `--radius` (10px),
  `--radius-sm` (6px), `--shadow`, `--font-sans` (системный стек), `--font-mono`.
- Светлая палитра по макету: фон `#f7f7f8`, поверхность `#ffffff`, рамка `#e6e6e9`, текст
  `#111114`, приглушённый `#6b6b76`, акцент `#111114` (основная кнопка тёмная), успех `#16a34a`,
  предупреждение `#d97706`, ошибка `#dc2626`, ссылка/выделение `#2563eb`.

**Topbar:** логотип и имя «Агент», поиск (заглушка с подсказкой ⌘K), индикатор соединения
(«Live» зелёный / «Нет связи»), выбор модели по умолчанию.

**LeftSidebar:** проекты (выбор, «+» открывает форму имени и пути), сессии выбранного проекта
деревом (родитель → дочерние), точка статуса (running — синяя пульсирующая, awaiting_confirmation
— жёлтая, completed — зелёная, error — красная, остальное — серая), архив свёрнут, внизу
«Настройки» и «Выйти».

**Центр:** заголовок сессии, режим, статус. Лента `ChatTimeline`. Внизу `Composer`.
`EmptyState` на `/`: иконка, «Что будем делать?», 4 карточки («Новая идея», «Составить план»,
«Разобрать код», «Найти ошибку»), клик вставляет шаблон текста в composer.

**RightPanel:** вкладки «Контекст», «Инструменты», «Агенты», «Аудит». Контекст — полоса
`approximate_tokens / context_window`, число пропущенных сообщений. Агенты — дочерние сессии.
Аудит — свёрнутый список исходящих запросов. Инструменты — пока список только для чтения.

## 7. Компоненты ленты

- **MessageItem.** Пользователь — справа, фон `--bg-subtle`; ассистент — слева, без фона.
  Текст через `MarkdownView`. Под сообщением ассистента — время и уровень рассуждений.
- **ReasoningBlock.** Свёрнутый блок «Рассуждения» с числом символов; раскрытие показывает текст
  приглушённым цветом. Во время стриминга — «Думаю…» с анимацией.
- **MarkdownView.** `markdown-it` (`html: false`, `linkify: true`) → `DOMPurify.sanitize` →
  `v-html`. Блоки кода подсвечиваются `shiki` асинхронно; кнопка «Копировать» у блока кода.
  Ссылки открываются с `rel="noopener noreferrer" target="_blank"`.
- **ToolCallCard.** Строка: иконка по инструменту, имя, краткий аргумент (`path`, `command` или
  `pattern`), статус-значок, раскрытие. Раскрытие: вкладки «Кратко» / «JSON». Для `shell` —
  моноширинный stdout/stderr и exit code; для `write_file`/`edit_file`/`multi_edit` — путь и
  изменения (old → new построчно, без внешней библиотеки diff).
- **ApprovalInline.** Внутри карточки при `awaiting_approval`: уровень риска, кнопки «Разрешить»
  и «Отклонить»; «Отклонить» раскрывает поле комментария. Горячие клавиши: `A` — разрешить,
  `R` — отклонить для первой ожидающей карточки, только когда фокус не в поле ввода.
- **Composer.** Textarea с автоувеличением до 10 строк. Enter — отправить, Shift+Enter — новая
  строка. Выбор режима (dev/ask) и модели для новой сессии; выбор уровня рассуждений
  (auto + уровни модели) для сообщения. Кнопка «Стоп» вместо «Отправить», пока сессия работает.
  Черновик сохраняется в `localStorage` по ключу сессии.

## 8. Тесты (минимум)

- `stores/session`: построение `timeline` (вызов + результат + подтверждение, skipped,
  summary), применение `message.delta`/`stream.reset`/`message.created`.
- `stores/app`: дерево сессий, применение `session.status`.
- `realtime`: переподключение с ростом задержки (фейковые таймеры).
- `MarkdownView`: `<script>` и `onerror` удаляются; `html: false` не пропускает сырой HTML.
- `ToolCallCard`: краткий аргумент для `shell`, `read_file`, `glob`; статус подтверждения.
- `Composer`: Enter отправляет, Shift+Enter — нет; кнопка «Стоп» при работающей сессии.

## 9. Вне 3.1–3.2

Кнопки под сообщениями (правка, перегенерация, ответвление) — после шага 2.4. Пользователи и
права — после фазы «Команда». Страница «Модели» — шаг 3.3. Explorer, интеграции и системная
панель макета — после основного каркаса.

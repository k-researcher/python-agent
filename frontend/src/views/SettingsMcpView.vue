<script setup lang="ts">
// MCP servers give the agent new tools. Commands run in the OS sandbox.
// Every tool call waits for the user's approval.

import { computed, onMounted, reactive, ref } from "vue";
import { ChevronDown, CircleAlert, Pencil, Plus, Trash2, TriangleAlert, X } from "lucide-vue-next";
import { api } from "../api/client";
import type { Project } from "../api/types";
import {
  MCP_ENV_KEY,
  MCP_SERVER_ID,
  joinCommand,
  mcpApi,
  splitCommand,
  type McpCheck,
  type McpDraft,
  type McpHttpProtocol,
  type McpList,
  type McpProfile,
  type McpServerItem,
} from "../api/mcp";
import SettingsDialog from "../components/settings/SettingsDialog.vue";
import SettingsHeader from "../components/settings/SettingsHeader.vue";
import ToggleSwitch from "../components/settings/ToggleSwitch.vue";
import { MODE_LABELS, SESSION_MODES } from "../modes";

const ROLE_PROFILES: Record<string, McpProfile> = {
  dev: "dev", pm: "dev", qa: "dev", devops: "dev", ask: "ask", arch: "ask", cybersec: "ask", analyst: "ask",
};
const PROFILES: { value: McpProfile; label: string; roles: string }[] = (["dev", "ask"] as const).map((value) => ({
  value,
  label: value === "dev" ? "Роли, которые меняют файлы" : "Роли только для чтения",
  roles: SESSION_MODES.filter((mode) => ROLE_PROFILES[mode] === value).map((mode) => MODE_LABELS[mode]).join(", "),
}));
const PROTOCOLS: { value: McpHttpProtocol; label: string }[] = [
  { value: "auto", label: "Авто" },
  { value: "modern", label: "Новая (2026-07-28)" },
  { value: "legacy", label: "С рукопожатием" },
];

const list = ref<McpList | null>(null);
const projects = ref<Project[]>([]);
const loadError = ref("");
const actionError = ref("");
const status = ref("");
const pending = ref("");
const checking = ref<Set<string>>(new Set());
const opened = ref<Set<string>>(new Set());

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function load(): Promise<void> {
  loadError.value = "";
  try {
    const [servers, all] = await Promise.all([mcpApi.list(), api.projects()]);
    list.value = servers;
    projects.value = all;
  } catch (error) {
    loadError.value = `Не удалось загрузить MCP-серверы: ${message(error)}`;
  }
}

async function change(key: string, action: () => Promise<McpList>, done = ""): Promise<boolean> {
  pending.value = key;
  actionError.value = "";
  status.value = "";
  try {
    list.value = await action();
    status.value = done;
    return true;
  } catch (error) {
    actionError.value = message(error);
    return false;
  } finally {
    pending.value = "";
  }
}

function projectNames(ids: string[]): string {
  if (!ids.length) return "Все проекты";
  return ids.map((id) => projects.value.find((item) => item.id === id)?.name ?? id).join(", ");
}

function profileText(modes: McpProfile[]): string {
  if (modes.length === 2) return "Все роли";
  return modes[0] === "dev" ? "Роли, которые меняют файлы" : "Роли только для чтения";
}

function plural(count: number, one: string, few: string, many: string): string {
  const tens = count % 100;
  const units = count % 10;
  if (tens >= 11 && tens <= 14) return many;
  if (units === 1) return one;
  if (units >= 2 && units <= 4) return few;
  return many;
}

function checkText(check: McpCheck): string {
  if (!check.ok) return check.error || "Сервер не запустился";
  const count = check.tool_count ?? check.tools?.length ?? 0;
  const name = check.server?.name ? ` · ${check.server.name}${check.server.version ? ` ${check.server.version}` : ""}` : "";
  return `Работает · ${count} ${plural(count, "инструмент", "инструмента", "инструментов")}${name}${check.protocol ? ` · ${check.protocol}` : ""}`;
}

function toggleDetails(id: string): void {
  const next = new Set(opened.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  opened.value = next;
}

async function check(item: McpServerItem): Promise<void> {
  checking.value = new Set([...checking.value, item.id]);
  actionError.value = "";
  status.value = "";
  try {
    list.value = await mcpApi.check(item.id);
    const result = list.value.servers.find((server) => server.id === item.id)?.last_check;
    if (result && (!result.ok || result.tools?.length)) opened.value = new Set([...opened.value, item.id]);
  } catch (error) {
    actionError.value = message(error);
  } finally {
    const next = new Set(checking.value);
    next.delete(item.id);
    checking.value = next;
  }
}

function setEnabled(item: McpServerItem, enabled: boolean): void {
  void change(`switch:${item.id}`, () => mcpApi.update(item.id, { enabled }));
}

// ---------- create and edit ----------

interface EnvRow { key: string; value: string; saved: boolean; removed: boolean }

const editor = reactive({
  open: false,
  mode: "create" as "create" | "edit",
  id: "",
  command: "",
  transport: "command" as "command" | "http",
  url: "",
  protocol: "auto" as McpHttpProtocol,
  env: [] as EnvRow[],
  allowNetwork: false,
  readPaths: "",
  allProjects: true,
  projectIds: [] as string[],
  profiles: ["dev"] as McpProfile[],
  saving: false,
  touched: false,
  submitted: false,
  error: "",
});

const idError = computed(() => {
  if (editor.mode === "edit") return "";
  if (!editor.id) return editor.touched ? "Введите ID сервера" : "";
  if (!MCP_SERVER_ID.test(editor.id)) return "Строчные латинские буквы, цифры, дефис и подчёркивание";
  if (list.value?.file_sources.includes(editor.id)) return "Такой ID уже у сервера из файла настроек";
  return list.value?.servers.some((item) => item.id === editor.id) ? "Сервер с таким ID уже есть" : "";
});

const commandParts = computed<string[] | string>(() => {
  try {
    return splitCommand(editor.command);
  } catch (error) {
    return message(error);
  }
});

const commandError = computed(() => {
  const parts = commandParts.value;
  if (typeof parts === "string") return parts;
  return !parts.length && editor.submitted ? "Введите команду запуска" : "";
});

const urlError = computed(() => {
  if (!editor.url) return editor.submitted ? "Введите адрес сервера" : "";
  try {
    const url = new URL(editor.url);
    if (!["https:", "http:"].includes(url.protocol) || !url.hostname) return "Укажите адрес HTTP или HTTPS";
    if (url.username || url.password) return "Добавьте ключ в заголовки, а не в адрес";
    if (url.hash || /[\s]/.test(editor.url)) return "Адрес не должен содержать пробелы и #";
  } catch {
    return "Укажите полный адрес сервера";
  }
  return "";
});

const envError = computed(() => {
  const http = editor.transport === "http";
  const names = editor.env.filter((row) => !row.removed).map((row) => row.key.trim());
  for (const name of names) {
    if (!name) return http ? "Введите имя заголовка" : "Введите имя переменной";
    if (!(http ? /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/ : MCP_ENV_KEY).test(name)) return `Недопустимое имя: ${name}`;
  }
  const fresh = editor.env.filter((row) => !row.removed && !row.saved && !row.value);
  if (fresh.length) return `Введите значение ${http ? "заголовка" : "переменной"} ${fresh[0].key || ""}`.trim();
  const unique = http ? names.map((name) => name.toLowerCase()) : names;
  return new Set(unique).size === unique.length ? "" : `Имена ${http ? "заголовков" : "переменных"} повторяются`;
});

const pathList = computed(() => editor.readPaths.split("\n").map((line) => line.trim()).filter(Boolean));
const pathsError = computed(() => {
  const wrong = pathList.value.find((path) => !path.startsWith("/"));
  return wrong ? `Путь должен быть абсолютным: ${wrong}` : "";
});

const projectsError = computed(() => (!editor.allProjects && !editor.projectIds.length ? "Выберите проекты" : ""));
const profilesError = computed(() => (editor.profiles.length ? "" : "Выберите хотя бы одну группу ролей"));

const canSave = computed(() => {
  const parts = commandParts.value;
  return (
    !editor.saving &&
    (editor.mode === "edit" || (!!editor.id && !idError.value)) &&
    (editor.transport === "http" ? !!editor.url && !urlError.value : Array.isArray(parts) && parts.length > 0) &&
    !envError.value &&
    (editor.transport === "http" || !pathsError.value) &&
    !projectsError.value &&
    !profilesError.value
  );
});

function openCreate(): void {
  Object.assign(editor, {
    open: true,
    mode: "create",
    id: "",
    command: "",
    transport: "command",
    url: "",
    protocol: "auto",
    env: [],
    allowNetwork: false,
    readPaths: "",
    allProjects: true,
    projectIds: [],
    profiles: ["dev"],
    saving: false,
    touched: false,
    submitted: false,
    error: "",
  });
}

function openEdit(item: McpServerItem): void {
  Object.assign(editor, {
    open: true,
    mode: "edit",
    id: item.id,
    command: joinCommand(item.command),
    transport: item.url ? "http" : "command",
    url: item.url ?? "",
    protocol: item.http_protocol,
    env: (item.url ? item.header_keys : item.env_keys).map((key) => ({ key, value: "", saved: true, removed: false })),
    allowNetwork: item.allow_network,
    readPaths: item.read_paths.join("\n"),
    allProjects: !item.project_ids.length,
    projectIds: [...item.project_ids],
    profiles: [...item.allowed_modes],
    saving: false,
    touched: false,
    submitted: false,
    error: "",
  });
}

function setTransport(transport: "command" | "http"): void {
  if (editor.mode === "edit" || editor.transport === transport) return;
  editor.transport = transport;
  editor.env = [];
}

function addEnv(): void {
  editor.env.push({ key: "", value: "", saved: false, removed: false });
}

function removeEnv(row: EnvRow): void {
  if (row.saved) row.removed = true;
  else editor.env.splice(editor.env.indexOf(row), 1);
}

function closeEditor(): void {
  if (!editor.saving) editor.open = false;
}

function draft(): McpDraft {
  const env: Record<string, string | null> = {};
  for (const row of editor.env) {
    if (row.removed) env[row.key] = null;
    else if (row.value) env[row.key.trim()] = row.value;
  }
  return {
    ...(editor.transport === "http" ? {
      url: editor.url,
      headers: env,
      http_protocol: editor.protocol,
    } : {
      command: commandParts.value as string[],
      env,
      allow_network: editor.allowNetwork,
      read_paths: editor.mode === "create" && !pathList.value.length ? null : pathList.value,
    }),
    project_ids: editor.allProjects ? [] : editor.projectIds,
    allowed_modes: editor.profiles,
  };
}

async function save(): Promise<void> {
  editor.touched = true;
  editor.submitted = true;
  if (!canSave.value) return;
  editor.saving = true;
  editor.error = "";
  try {
    list.value = editor.mode === "create" ? await mcpApi.create(editor.id, draft()) : await mcpApi.update(editor.id, draft());
    status.value = editor.mode === "create"
      ? `Сервер «${editor.id}» добавлен. Нажмите «Проверить», чтобы запустить его и увидеть инструменты.`
      : `Сервер «${editor.id}» сохранён`;
    editor.open = false;
  } catch (error) {
    editor.error = message(error);
  } finally {
    editor.saving = false;
  }
}

// ---------- delete ----------

const removing = ref<McpServerItem | null>(null);
const removeError = ref("");

function askRemove(item: McpServerItem): void {
  removeError.value = "";
  removing.value = item;
}

async function remove(): Promise<void> {
  const item = removing.value;
  if (!item) return;
  pending.value = `remove:${item.id}`;
  removeError.value = "";
  try {
    list.value = await mcpApi.remove(item.id);
    status.value = `Сервер «${item.id}» удалён`;
    removing.value = null;
  } catch (error) {
    removeError.value = message(error);
  } finally {
    pending.value = "";
  }
}

onMounted(load);
</script>

<template>
  <section class="ui-page" data-testid="mcp-settings">
    <SettingsHeader title="MCP" />

    <p class="ui-hint">
      MCP-сервер даёт агенту новые инструменты: поиск по документации, работу с сервисами и другое.
      Можно запустить программу или подключиться по адресу HTTP. Программа работает в песочнице:
      видит папку проекта и указанные папки, а сеть получает, только если её разрешить.
      Каждый вызов инструмента агент просит подтвердить.
    </p>

    <p v-if="loadError" class="ui-notice ui-notice--error" role="alert">
      <CircleAlert :size="16" aria-hidden="true" />
      <span class="ui-notice__text">{{ loadError }}</span>
      <button type="button" class="ui-link" @click="load">Повторить</button>
    </p>

    <template v-if="list">
      <p v-if="!list.sandbox && (editor.open ? editor.transport === 'command' : list.servers.some(item => !item.url))" class="ui-notice ui-notice--warning" data-testid="mcp-no-sandbox">
        <TriangleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">
          Песочница ОС недоступна на этом компьютере, поэтому MCP-серверы с командой не запустятся. Настройки можно сохранить заранее.
        </span>
      </p>
      <div aria-live="polite" role="status" class="live">
        <p v-if="status" class="ui-notice ui-notice--success" data-testid="mcp-status">
          <span class="ui-notice__text">{{ status }}</span>
          <button type="button" class="ui-btn ui-btn--icon" aria-label="Скрыть сообщение" @click="status = ''">
            <X :size="15" aria-hidden="true" />
          </button>
        </p>
      </div>
      <p v-if="actionError" class="ui-notice ui-notice--error" role="alert" data-testid="mcp-error">
        <CircleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">{{ actionError }}</span>
        <button type="button" class="ui-btn ui-btn--icon" aria-label="Скрыть сообщение" @click="actionError = ''">
          <X :size="15" aria-hidden="true" />
        </button>
      </p>

      <section class="ui-section" aria-labelledby="mcp-title">
        <div class="ui-section__head">
          <h2 id="mcp-title" class="ui-section__title">Серверы</h2>
          <div class="ui-section__actions">
            <button type="button" class="ui-btn ui-btn--filled" data-testid="mcp-add" @click="openCreate">
              <Plus :size="16" aria-hidden="true" />
              <span>Добавить сервер</span>
            </button>
          </div>
        </div>
        <p class="ui-hint">
          Выключатель включает сервер для новых запусков агента. «Проверить» запускает сервер и показывает его
          инструменты.
        </p>

        <div v-if="!list.servers.length" class="ui-empty" data-testid="mcp-empty">
          <p>MCP-серверов пока нет.</p>
          <p class="empty__more">Добавьте сервер: укажите команду запуска или адрес HTTP.</p>
        </div>

        <article
          v-for="item in list.servers"
          :key="item.id"
          class="ui-card"
          :class="{ 'ui-card--off': !item.enabled }"
          :data-server="item.id"
        >
          <div class="ui-card__row">
            <div class="ui-card__main">
              <strong class="ui-card__title">{{ item.id }} <span v-if="item.url" class="ui-badge">HTTP</span></strong>
              <span class="mono ui-card__meta command">{{ item.url || joinCommand(item.command) }}</span>
              <span class="ui-card__meta">
                {{ projectNames(item.project_ids) }} · {{ profileText(item.allowed_modes) }}
                <template v-if="!item.url"> · {{ item.allow_network ? "сеть разрешена" : "без сети" }}</template>
              </span>
              <span v-if="item.env_keys.length" class="ui-card__meta">
                Секреты: <span class="mono">{{ item.env_keys.join(", ") }}</span>
              </span>
              <span v-if="item.header_keys?.length" class="ui-card__meta">
                Заголовки: <span class="mono">{{ item.header_keys.join(", ") }}</span>
              </span>
              <span v-if="checking.has(item.id)" class="ui-card__meta" data-testid="mcp-check">Запускаю сервер…</span>
              <button
                v-else-if="item.last_check"
                type="button"
                class="check-line ui-small"
                :class="item.last_check.ok ? 'ui-ok' : 'ui-bad'"
                :aria-expanded="opened.has(item.id)"
                :aria-controls="`mcp-details-${item.id}`"
                data-testid="mcp-check"
                @click="toggleDetails(item.id)"
              >
                <span>{{ checkText(item.last_check) }}</span>
                <ChevronDown :size="14" aria-hidden="true" class="check-line__chevron" />
              </button>
              <span v-else class="ui-card__meta">Ещё не проверялся</span>
            </div>
            <div class="ui-card__actions">
              <button
                type="button"
                class="ui-btn ui-btn--outlined"
                :disabled="checking.has(item.id)"
                data-testid="mcp-check-run"
                @click="check(item)"
              >{{ checking.has(item.id) ? "Проверяю…" : "Проверить" }}</button>
              <button
                type="button"
                class="ui-btn ui-btn--icon"
                :aria-label="`Изменить сервер ${item.id}`"
                title="Изменить"
                data-testid="mcp-edit"
                @click="openEdit(item)"
              >
                <Pencil :size="16" aria-hidden="true" />
              </button>
              <button
                type="button"
                class="ui-btn ui-btn--icon ui-btn--danger"
                :aria-label="`Удалить сервер ${item.id}`"
                title="Удалить"
                data-testid="mcp-remove"
                @click="askRemove(item)"
              >
                <Trash2 :size="16" aria-hidden="true" />
              </button>
              <ToggleSwitch
                :model-value="item.enabled"
                :label="`Сервер ${item.id} включён`"
                :disabled="pending === `switch:${item.id}`"
                data-testid="mcp-switch"
                @update:model-value="setEnabled(item, $event)"
              />
            </div>
          </div>
          <div
            v-if="item.last_check && opened.has(item.id) && !checking.has(item.id)"
            :id="`mcp-details-${item.id}`"
            class="details"
            data-testid="mcp-details"
          >
            <ul v-if="item.last_check.ok && item.last_check.tools?.length" class="tools">
              <li v-for="tool in item.last_check.tools" :key="tool.name">
                <span class="mono">{{ tool.name }}</span>
                <span v-if="tool.description" class="ui-card__meta">{{ tool.description }}</span>
              </li>
            </ul>
            <p v-else-if="item.last_check.ok" class="ui-card__meta">Сервер не сообщил ни одного инструмента.</p>
            <template v-else>
              <p class="ui-card__meta">
                <template v-if="item.url">Проверьте адрес, заголовки доступа и выбранный протокол.</template>
                <template v-else>Проверьте команду в терминале. Если программе не хватает папки, добавьте её в «Папки для чтения».</template>
              </p>
              <pre v-if="item.last_check.stderr" class="stderr">{{ item.last_check.stderr }}</pre>
            </template>
          </div>
        </article>

        <p v-if="list.file_sources.length" class="ui-hint file-sources" data-testid="mcp-file-sources">
          Из файла настроек: <span class="mono">{{ list.file_sources.join(", ") }}</span>. Их меняют в файле.
        </p>
      </section>
    </template>
    <p v-else-if="!loadError" class="ui-hint">Загрузка…</p>

    <SettingsDialog
      :open="editor.open"
      :title="editor.mode === 'create' ? 'Новый MCP-сервер' : `Сервер ${editor.id}`"
      wide
      @close="closeEditor"
    >
      <form id="mcp-form" class="ui-form" novalidate data-testid="mcp-form" @submit.prevent="save">
        <label v-if="editor.mode === 'create'" class="ui-field">
          <span class="ui-field__label">ID</span>
          <input
            v-model.trim="editor.id"
            class="ui-input mono"
            :aria-invalid="!!idError"
            aria-describedby="mcp-id-help"
            autocomplete="off"
            spellcheck="false"
            data-autofocus
            data-testid="mcp-id"
            @blur="editor.touched = true"
          />
          <small id="mcp-id-help" class="ui-field__help" :class="{ 'ui-field__help--error': idError }">
            {{ idError || "Короткое имя, например docs. По нему названы инструменты сервера." }}
          </small>
        </label>
        <div class="ui-tabs" role="group" aria-label="Подключение">
          <button v-for="option in ([{ value: 'command', label: 'Команда' }, { value: 'http', label: 'Адрес' }] as const)"
            :key="option.value" type="button" role="radio" :aria-checked="editor.transport === option.value"
            :disabled="editor.mode === 'edit'" @click="setTransport(option.value)">{{ option.label }}</button>
        </div>
        <label v-if="editor.transport === 'http'" class="ui-field">
          <span class="ui-field__label">Адрес сервера</span>
          <input v-model.trim="editor.url" class="ui-input mono" type="url" autocomplete="off" spellcheck="false"
            :aria-invalid="!!urlError" aria-describedby="mcp-url-help" placeholder="https://example.com/mcp"
            :data-autofocus="editor.mode === 'edit' ? '' : undefined" data-testid="mcp-url" />
          <small id="mcp-url-help" class="ui-field__help" :class="{ 'ui-field__help--error': urlError }">
            {{ urlError || "HTTPS; HTTP разрешён только на этом компьютере. Ключи добавьте в заголовки." }}
          </small>
        </label>
        <div v-if="editor.transport === 'http'" class="ui-field">
          <span class="ui-field__label">Протокол</span>
          <div class="ui-tabs" role="group" aria-label="Протокол HTTP">
            <button v-for="option in PROTOCOLS" :key="option.value" type="button" role="radio"
              :aria-checked="editor.protocol === option.value" @click="editor.protocol = option.value">{{ option.label }}</button>
          </div>
        </div>
        <label v-else class="ui-field">
          <span class="ui-field__label">Команда запуска</span>
          <input
            v-model="editor.command"
            class="ui-input mono"
            :aria-invalid="!!commandError"
            aria-describedby="mcp-command-help"
            autocomplete="off"
            spellcheck="false"
            placeholder="npx -y пакет-сервера"
            :data-autofocus="editor.mode === 'edit' ? '' : undefined"
            data-testid="mcp-command"
          />
          <small id="mcp-command-help" class="ui-field__help" :class="{ 'ui-field__help--error': commandError }">
            {{ commandError || "Как в терминале. Часть с пробелами возьмите в кавычки." }}
          </small>
        </label>

        <fieldset class="group">
          <legend class="ui-field__label">{{ editor.transport === 'http' ? 'Заголовки' : 'Секреты' }}</legend>
          <p class="ui-field__help">
            <span>Ключи и токены для сервера. Хранятся зашифрованными и больше не показываются.</span>
          </p>
          <div v-for="(row, index) in editor.env" v-show="!row.removed" :key="index" class="env-row" data-testid="mcp-env-row">
            <input
              v-model="row.key"
              class="ui-input mono"
              :disabled="row.saved"
              :aria-label="`Имя ${editor.transport === 'http' ? 'заголовка' : 'переменной'} ${index + 1}`"
              placeholder="ИМЯ"
              autocomplete="off"
              spellcheck="false"
              data-testid="mcp-env-key"
            />
            <input
              v-model="row.value"
              class="ui-input"
              type="password"
              :aria-label="`Значение ${editor.transport === 'http' ? 'заголовка' : 'переменной'} ${row.key || index + 1}`"
              :placeholder="row.saved ? 'Сохранено — введите, чтобы заменить' : 'Значение'"
              autocomplete="new-password"
              data-testid="mcp-env-value"
            />
            <button
              type="button"
              class="ui-btn ui-btn--icon ui-btn--danger"
              :aria-label="`Убрать ${editor.transport === 'http' ? 'заголовок' : 'переменную'} ${row.key}`"
              title="Убрать"
              @click="removeEnv(row)"
            >
              <X :size="16" aria-hidden="true" />
            </button>
          </div>
          <button type="button" class="ui-btn ui-btn--text add-env" data-testid="mcp-env-add" @click="addEnv">
            <Plus :size="16" aria-hidden="true" />
            <span>{{ editor.transport === 'http' ? 'Добавить заголовок' : 'Добавить секрет' }}</span>
          </button>
          <small v-if="envError" class="ui-field__help ui-field__help--error">{{ envError }}</small>
        </fieldset>

        <div v-if="editor.transport === 'command'" class="switch-row">
          <div>
            <span class="ui-field__label">Доступ к сети</span>
            <p class="ui-field__help">
              <span>Нужен серверу, который ходит в интернет или скачивает пакеты, например через npx.</span>
            </p>
          </div>
          <ToggleSwitch v-model="editor.allowNetwork" label="Доступ к сети" data-testid="mcp-network" />
        </div>

        <label v-if="editor.transport === 'command'" class="ui-field">
          <span class="ui-field__label">Папки для чтения</span>
          <textarea
            v-model="editor.readPaths"
            class="ui-input mono"
            rows="2"
            :aria-invalid="!!pathsError"
            spellcheck="false"
            :placeholder="editor.mode === 'create' ? 'Пусто — папка программы определится сама' : '/opt/homebrew'"
            data-testid="mcp-paths"
          />
          <small class="ui-field__help" :class="{ 'ui-field__help--error': pathsError }">
            {{ pathsError || "По одной на строку. Программа сервера и её среда должны быть в этих папках." }}
          </small>
        </label>

        <fieldset class="group">
          <legend class="ui-field__label">Проекты</legend>
          <label class="ui-choice">
            <input v-model="editor.allProjects" type="radio" :value="true" data-testid="mcp-all-projects" />
            <span>Все проекты</span>
          </label>
          <label class="ui-choice">
            <input v-model="editor.allProjects" type="radio" :value="false" :disabled="!projects.length" />
            <span>Только выбранные</span>
          </label>
          <div v-if="!editor.allProjects" class="choices">
            <label v-for="project in projects" :key="project.id" class="ui-choice">
              <input v-model="editor.projectIds" type="checkbox" :value="project.id" :data-project="project.id" />
              <span>{{ project.name }}</span>
            </label>
          </div>
          <small v-if="projectsError" class="ui-field__help ui-field__help--error">{{ projectsError }}</small>
        </fieldset>

        <fieldset class="group">
          <legend class="ui-field__label">Роли агента</legend>
          <label v-for="profile in PROFILES" :key="profile.value" class="ui-choice profile">
            <input v-model="editor.profiles" type="checkbox" :value="profile.value" :data-profile="profile.value" />
            <span>
              {{ profile.label }}
              <small class="ui-field__help"><span>{{ profile.roles }}</span></small>
            </span>
          </label>
          <small v-if="profilesError" class="ui-field__help ui-field__help--error">{{ profilesError }}</small>
        </fieldset>

        <p v-if="editor.error" class="ui-notice ui-notice--error" role="alert" data-testid="mcp-form-error">
          <CircleAlert :size="16" aria-hidden="true" />
          <span class="ui-notice__text">{{ editor.error }}</span>
        </p>
      </form>
      <template #actions>
        <button type="button" class="ui-btn ui-btn--text" :disabled="editor.saving" @click="closeEditor">Отмена</button>
        <button type="submit" form="mcp-form" class="ui-btn ui-btn--filled" :disabled="!canSave" data-testid="mcp-save">
          {{ editor.saving ? "Сохраняю…" : editor.mode === "create" ? "Добавить" : "Сохранить" }}
        </button>
      </template>
    </SettingsDialog>

    <SettingsDialog
      :open="!!removing"
      :title="removing ? `Удалить сервер «${removing.id}»?` : ''"
      alert
      @close="pending.startsWith('remove:') || (removing = null)"
    >
      <p>Агент больше не получит инструменты этого сервера. Сохранённые секреты удалятся. Программа сервера не меняется.</p>
      <p v-if="removeError" class="ui-notice ui-notice--error" role="alert">
        <CircleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">{{ removeError }}</span>
      </p>
      <template #actions>
        <button
          type="button"
          class="ui-btn ui-btn--text"
          data-autofocus
          :disabled="pending.startsWith('remove:')"
          @click="removing = null"
        >Отмена</button>
        <button
          type="button"
          class="ui-btn ui-btn--filled ui-btn--danger"
          :disabled="pending.startsWith('remove:')"
          data-testid="mcp-remove-confirm"
          @click="remove"
        >{{ pending.startsWith("remove:") ? "Удаляю…" : "Удалить" }}</button>
      </template>
    </SettingsDialog>
  </section>
</template>

<style scoped>
.live:empty {
  display: none;
}

.ui-card {
  margin-top: 8px;
}

.command {
  overflow-wrap: anywhere;
}

/* The check result opens the list of tools or the error output. */
.check-line {
  display: inline-flex;
  align-self: flex-start;
  align-items: center;
  gap: 4px;
  margin-top: 2px;
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}

.check-line__chevron {
  flex: none;
  transition: transform 0.15s ease;
}

.check-line[aria-expanded="true"] .check-line__chevron {
  transform: rotate(180deg);
}

.details {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}

.tools {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tools li {
  display: flex;
  flex-direction: column;
  gap: 1px;
  font-size: 13px;
  overflow-wrap: anywhere;
}

.stderr {
  max-height: 220px;
  margin-top: 8px;
  padding: 10px 12px;
  overflow: auto;
  border-radius: var(--radius);
  background: var(--code-bg);
  color: var(--code-text);
  font-size: 12px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.empty__more {
  margin-top: 4px;
  font-size: 13px;
}

.file-sources {
  margin-top: 16px;
}

.group {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  border: 0;
}

.group legend {
  margin-bottom: 2px;
}

.env-row {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(0, 3fr) auto;
  gap: 8px;
  align-items: center;
}

.add-env {
  align-self: flex-start;
  margin-left: -12px;
}

.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.choices {
  display: flex;
  flex-direction: column;
  padding-left: 26px;
}

.profile {
  align-items: flex-start;
}

.profile input {
  margin-top: 4px;
}

.profile .ui-field__help {
  display: block;
}

@media (prefers-reduced-motion: reduce) {
  .check-line__chevron {
    transition: none;
  }
}

@media (max-width: 599px) {
  .env-row {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .env-row input:first-child {
    grid-column: 1 / -1;
  }
}
</style>

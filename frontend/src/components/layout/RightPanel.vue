<script setup lang="ts">
// Right panel: tabs "Контекст", "Инструменты", "Агенты", "Аудит".
// Presentational — data via props, opening a child session goes up through emits.

import { computed, nextTick, ref } from "vue";
import { Bot, ClipboardList, Puzzle, Activity, ChevronRight } from "lucide-vue-next";
import type { ContextStats } from "./types";

const props = withDefaults(
  defineProps<{
    context: ContextStats | null;
    children: { id: string; title: string; status: string }[];
    audit: { id: number; operation: string; destination: string; status: string; created_at: string }[];
    tools: { name: string; enabled: boolean }[];
  }>(),
  {
    context: null,
    children: () => [],
    audit: () => [],
    tools: () => [],
  },
);

const emit = defineEmits<{
  openSession: [id: string];
}>();

const TAB_CONTEXT = "context";
const TAB_TOOLS = "tools";
const TAB_AGENTS = "agents";
const TAB_AUDIT = "audit";

const tabs = [
  { id: TAB_CONTEXT, label: "Контекст" },
  { id: TAB_TOOLS, label: "Инструменты" },
  { id: TAB_AGENTS, label: "Агенты" },
  { id: TAB_AUDIT, label: "Аудит" },
] as const;

type TabId = (typeof tabs)[number]["id"];

const activeTab = ref<TabId>(TAB_CONTEXT);

/**
 * Context meter denominator: prefer a positive finite input_budget (newer
 * server replies), fall back to context_window (older server). Guards against
 * missing/NaN/Infinity/negative values from the API.
 */
const denominator = computed<number>(() => {
  const stats = props.context;
  if (!stats) return 0;
  if (
    typeof stats.input_budget === "number" &&
    Number.isFinite(stats.input_budget) &&
    stats.input_budget > 0
  ) {
    return stats.input_budget;
  }
  if (
    typeof stats.context_window === "number" &&
    Number.isFinite(stats.context_window) &&
    stats.context_window > 0
  ) {
    return stats.context_window;
  }
  return 0;
});

/** Context fill 0–100 %, clamped; 0 when stats are missing or unmeasurable. */
const contextPercent = computed(() => {
  const stats = props.context;
  const window = denominator.value;
  if (
    !stats ||
    window <= 0 ||
    typeof stats.approximate_tokens !== "number" ||
    !Number.isFinite(stats.approximate_tokens) ||
    stats.approximate_tokens < 0
  ) {
    return 0;
  }
  const ratio = stats.approximate_tokens / window;
  return Math.min(100, Math.round(ratio * 100));
});

const contextTone = computed(() => (contextPercent.value >= 80 ? "--warn" : ""));

const stageLabel = computed(() => {
  const stage = props.context?.stage;
  if (!stage || stage === "none") return "";
  const labels: Record<string, string> = {
    stage_1: "Сокращены результаты инструментов",
    stage_2: "Свёрнуты вызовы инструментов",
    summary: "Старые сообщения заменены сводкой",
    summarize: "Старые сообщения заменены сводкой",
    deterministic: "История сокращена без модели",
  };
  return Object.hasOwn(labels, stage) ? labels[stage] : "";
});

/** Negative omitted_messages is treated as 0 (corrupt data). */
const omittedCount = computed(() => {
  const value = props.context?.omitted_messages;
  if (value === undefined || value === null) return 0;
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
});

function formatTokens(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value) || value < 0) return "—";
  return value.toLocaleString("ru-RU");
}

function onChildClick(id: string): void {
  emit("openSession", id);
}

function focusTab(id: TabId): void {
  activeTab.value = id;
  nextTick(() => document.getElementById(`tab-${id}`)?.focus());
}

function onTabKeydown(event: KeyboardEvent, tabId: TabId): void {
  const index = tabs.findIndex((t) => t.id === tabId);
  let next: TabId | null = null;
  switch (event.key) {
    case "ArrowRight":
      event.preventDefault();
      next = tabs[(index + 1) % tabs.length].id;
      break;
    case "ArrowLeft":
      event.preventDefault();
      next = tabs[(index - 1 + tabs.length) % tabs.length].id;
      break;
    case "Home":
      event.preventDefault();
      next = tabs[0].id;
      break;
    case "End":
      event.preventDefault();
      next = tabs[tabs.length - 1].id;
      break;
    default:
      return;
  }
  focusTab(next);
}
</script>

<template>
  <div class="right-panel">
    <div class="right-panel__tabs" role="tablist" aria-label="Панель">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :id="`tab-${tab.id}`"
        type="button"
        role="tab"
        class="right-panel__tab"
        :class="{ 'right-panel__tab--active': activeTab === tab.id }"
        :aria-selected="activeTab === tab.id"
        :aria-controls="`panel-${tab.id}`"
        :tabindex="activeTab === tab.id ? 0 : -1"
        :data-tab="tab.id"
        @click="activeTab = tab.id"
        @keydown="onTabKeydown($event, tab.id)"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- Контекст -->
    <section
      v-if="activeTab === TAB_CONTEXT"
      id="panel-context"
      role="tabpanel"
      aria-labelledby="tab-context"
      class="right-panel__panel"
      data-panel="context"
    >
      <template v-if="context">
        <div v-if="context.fits === false" class="context-block__error" data-context-error role="alert">
          <span v-if="context.error">{{ context.error }}</span>
          <span v-else>Контекст не помещается — часть данных не передана модели</span>
        </div>
        <div class="context-block">
          <div class="context-block__row">
            <span class="muted">Использовано контекста</span>
            <span class="mono context-block__percent" data-context-percent>{{ contextPercent }}%</span>
          </div>
          <div
            class="context-bar"
            role="progressbar"
            :aria-valuenow="contextPercent"
            aria-valuemin="0"
            aria-valuemax="100"
            :class="`context-bar${contextTone}`"
            :data-tone="contextTone || 'normal'"
          >
            <div class="context-bar__fill" :style="{ width: `${contextPercent}%` }" />
          </div>
          <p class="context-block__detail" data-context-detail>
            {{ formatTokens(context.approximate_tokens) }} из
            {{ formatTokens(denominator) }} токенов
          </p>
          <dl v-if="stageLabel || context.summary_used != null || context.would_summarize != null || context.output_tokens != null || context.safety_margin != null" class="context-block__meta">
            <template v-if="stageLabel">
              <div class="context-block__compression"><dt>Сжатие контекста</dt><dd data-context-stage>{{ stageLabel }}</dd></div>
            </template>
            <template v-if="context.summary_used !== undefined">
              <div><dt>Сводка</dt><dd>{{ context.summary_used ? "использована" : "не использована" }}</dd></div>
            </template>
            <template v-if="context.would_summarize !== undefined">
              <div><dt>Будет свёрнуто</dt><dd>{{ context.would_summarize ? "да" : "нет" }}</dd></div>
            </template>
            <template v-if="context.output_tokens !== undefined">
              <div><dt>Выходные токены</dt><dd class="mono">{{ formatTokens(context.output_tokens) }}</dd></div>
            </template>
            <template v-if="context.safety_margin !== undefined">
              <div><dt>Запас</dt><dd class="mono">{{ formatTokens(context.safety_margin) }}</dd></div>
            </template>
          </dl>
        </div>
        <p v-if="omittedCount > 0" class="context-block__omitted" data-omitted>
          Пропущено сообщений: {{ omittedCount }}
        </p>
        <p v-else class="muted context-block__omitted" data-omitted>Пропущенных сообщений нет</p>
      </template>
      <p v-else class="muted right-panel__empty" data-context-empty>Нет данных о контексте</p>
    </section>

    <!-- Инструменты -->
    <section
      v-if="activeTab === TAB_TOOLS"
      id="panel-tools"
      role="tabpanel"
      aria-labelledby="tab-tools"
      class="right-panel__panel"
      data-panel="tools"
    >
      <ul class="tool-list">
        <li v-for="tool in tools" :key="tool.name" class="tool-row" data-tool>
          <Puzzle :size="14" aria-hidden="true" class="tool-row__icon" />
          <span class="tool-row__name mono">{{ tool.name }}</span>
          <span
            class="tool-row__state"
            :class="tool.enabled ? 'tool-row__state--on' : 'tool-row__state--off'"
            :data-enabled="tool.enabled ? 'on' : 'off'"
          >
            {{ tool.enabled ? "вкл" : "выкл" }}
          </span>
        </li>
      </ul>
      <p v-if="tools.length === 0" class="muted right-panel__empty">Инструменты не настроены</p>
    </section>

    <!-- Агенты -->
    <section
      v-if="activeTab === TAB_AGENTS"
      id="panel-agents"
      role="tabpanel"
      aria-labelledby="tab-agents"
      class="right-panel__panel"
      data-panel="agents"
    >
      <ul class="agent-list">
        <li v-for="child in children" :key="child.id" class="agent-row">
          <button
            type="button"
            class="agent-row__button"
            :data-agent-id="child.id"
            @click="onChildClick(child.id)"
          >
            <Bot :size="15" aria-hidden="true" class="agent-row__icon" />
            <span class="agent-row__title">{{ child.title }}</span>
            <span class="muted agent-row__status" data-agent-status>{{ child.status }}</span>
            <ChevronRight :size="14" aria-hidden="true" class="agent-row__arrow" />
          </button>
        </li>
      </ul>
      <p v-if="children.length === 0" class="muted right-panel__empty">Нет дочерних агентов</p>
    </section>

    <!-- Аудит -->
    <section
      v-if="activeTab === TAB_AUDIT"
      id="panel-audit"
      role="tabpanel"
      aria-labelledby="tab-audit"
      class="right-panel__panel"
      data-panel="audit"
    >
      <ul class="audit-list">
        <li v-for="entry in audit" :key="entry.id" class="audit-row" data-audit>
          <Activity :size="14" aria-hidden="true" class="audit-row__icon" />
          <div class="audit-row__body">
            <div class="audit-row__top">
              <span class="audit-row__operation mono">{{ entry.operation }}</span>
              <span class="audit-row__status" data-audit-status>{{ entry.status }}</span>
            </div>
            <div class="audit-row__dest muted mono">{{ entry.destination }}</div>
            <div class="audit-row__time muted">{{ entry.created_at }}</div>
          </div>
        </li>
      </ul>
      <p v-if="audit.length === 0" class="muted right-panel__empty">Записей аудита нет</p>
    </section>
  </div>
</template>

<style scoped>
.right-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.right-panel__tabs {
  display: flex;
  gap: 0;
  padding: 8px 8px 0;
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
  scrollbar-width: thin;
}

.right-panel__tab {
  flex: 0 0 auto;
  padding: 7px 8px;
  border: 0;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: var(--text-muted);
  font-size: 13px;
  white-space: nowrap;
  cursor: pointer;
}

.right-panel__tab:hover {
  color: var(--text);
}

.right-panel__tab--active {
  color: var(--text);
  border-bottom-color: var(--accent);
}

.right-panel__panel {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 12px;
}

.context-block__row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 8px;
}

.context-block__percent {
  font-weight: 650;
}

.context-bar {
  height: 8px;
  border-radius: 999px;
  background: var(--bg-subtle);
  overflow: hidden;
}

.context-bar__fill {
  height: 100%;
  border-radius: 999px;
  background: var(--success);
  transition: width 0.2s ease;
}

.context-bar--warn .context-bar__fill {
  background: var(--warning);
}

.context-block__detail {
  margin-top: 8px;
  font-size: 13px;
  color: var(--text);
}

.context-block__error {
  margin-bottom: 10px;
  padding: 8px 10px;
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--danger) 10%, transparent);
  color: var(--danger);
  font-size: 13px;
}

.context-block__meta {
  display: grid;
  gap: 4px;
  margin: 10px 0 0;
  padding: 0;
}

.context-block__meta > div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
}

.context-block__meta dt {
  color: var(--text-muted);
}

.context-block__meta dd {
  margin: 0;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.context-block__meta > .context-block__compression {
  flex-wrap: wrap;
  gap: 4px 10px;
}

.context-block__compression dt {
  flex-shrink: 0;
  white-space: nowrap;
}

.context-block__compression dd {
  min-width: 0;
  max-width: 100%;
  overflow: visible;
  text-overflow: clip;
  white-space: normal;
  overflow-wrap: break-word;
}

.context-block__omitted {
  margin-top: 10px;
  font-size: 13px;
  color: var(--warning);
}

.right-panel__empty {
  padding: 12px 4px;
  font-size: 13px;
}

.tool-list,
.agent-list,
.audit-list {
  display: grid;
  gap: 4px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.tool-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 8px;
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
}

.tool-row__icon {
  color: var(--text-muted);
}

.tool-row__name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}

.tool-row__state {
  font-size: 12px;
  padding: 1px 7px;
  border-radius: 999px;
}

.tool-row__state--on {
  color: var(--success);
  background: color-mix(in srgb, var(--success) 14%, transparent);
}

.tool-row__state--off {
  color: var(--text-muted);
  background: var(--bg-subtle);
}

.agent-row__button {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: 0;
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  color: var(--text);
  text-align: left;
}

.agent-row__button:hover {
  background: var(--border);
}

.agent-row__icon {
  flex: none;
  color: var(--text-muted);
}

.agent-row__title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.agent-row__status {
  flex: none;
  font-size: 12px;
}

.agent-row__arrow {
  flex: none;
  color: var(--text-muted);
}

.audit-row {
  display: flex;
  gap: 8px;
  padding: 7px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.audit-row__icon {
  flex: none;
  margin-top: 2px;
  color: var(--text-muted);
}

.audit-row__body {
  min-width: 0;
  flex: 1;
}

.audit-row__top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.audit-row__operation {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.audit-row__status {
  flex: none;
  font-size: 12px;
  color: var(--text-muted);
  text-transform: lowercase;
}

.audit-row__dest,
.audit-row__time {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 1100px) {
  .right-panel__tab {
    font-size: 12px;
    padding: 7px 2px;
  }
}
</style>

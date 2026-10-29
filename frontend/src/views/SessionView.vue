<script setup lang="ts">
// Session detail view (/s/:id) — header, ChatTimeline, Composer.
// Owns the session lifecycle: opens the store on mount/id change,
// forwards send/stop to the session store, converts store timeline
// items to the chat component contract via typed adapters.

import { computed, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useSessionStore } from "../stores/session";
import { useAppStore } from "../stores/app";
import ChatTimeline from "../components/chat/ChatTimeline.vue";
import Composer from "../components/chat/Composer.vue";
import AgentSettings from "../components/chat/AgentSettings.vue";
import type { RejectPayload, SendPayload } from "../components/chat/types";
import { toChatTimelineItems } from "../uiAdapters";
import { sessionStatusLabel } from "../components/layout/status";
import { modeLabel as resolveModeLabel } from "../modes";

const props = defineProps<{ id: string }>();
const route = useRoute();
const router = useRouter();
const sessionStore = useSessionStore();
const appStore = useAppStore();

const isOpening = ref(true);
const openError = ref("");
const sendError = ref("");
const isSending = ref(false);
const selectedEffort = ref("auto");

let openSeq = 0;

// Draft storage key per session; becomes a fresh key after a failed send so the
// Composer remounts/re-reads the restored text.
const draftKey = ref(`draft:${props.id}`);
let draftRetryCounter = 0;

function restoreDraft(text: string): void {
  draftRetryCounter += 1;
  const key = `draft:${props.id}:r${draftRetryCounter}`;
  try {
    window.localStorage.setItem(key, text);
  } catch {
    // storage unavailable — the composer just starts empty
  }
  draftKey.value = key;
}

function clearDraft(): void {
  try {
    window.localStorage.removeItem(draftKey.value);
  } catch {
    // ignore storage write failures
  }
}

watch(
  () => props.id,
  (id) => {
    if (id) {
      draftKey.value = `draft:${id}`;
      draftRetryCounter = 0;
    }
  },
);

// Session header data
const session = computed(() => sessionStore.session);
const timelineItems = computed(() => toChatTimelineItems(sessionStore.timeline));
const external = computed(() => !!session.value?.engine_profile);
const modelLabel = computed(() => {
  if (external.value) return `Внешний агент · ${session.value?.engine_profile}`;
  const profile = appStore.config?.llm_profiles.find((model) => model.name === session.value?.llm_profile);
  return profile ? `${profile.name} · ${profile.model}` : session.value?.llm_profile ?? "По умолчанию";
});
const modeLabel = computed(() => resolveModeLabel(session.value?.mode));
const engineHealthLabel = computed(() => {
  const health = session.value?.engine_state?.health;
  if (health?.diagnostic) return health.diagnostic;
  if (["available", "ready"].includes(health?.status ?? "")) return "Внешний агент доступен";
  if (health?.status === "unknown") return "Состояние внешнего агента не проверено";
  return "engine недоступен";
});
const engineUsage = computed(() => session.value?.engine_state?.usage);
const modelSummary = computed(() => external.value ? modelLabel.value : appStore.config?.llm_profiles.find((model) => model.name === session.value?.llm_profile)?.model ?? session.value?.llm_profile ?? "По умолчанию");

const busy = computed(() => {
  if (isSending.value) return true;
  const status = session.value?.status;
  if (!status) return false;
  return ["pending", "running", "awaiting_confirmation", "starting"].includes(status);
});

const efforts = computed<string[]>(() => {
  if (external.value) return [];
  const profile = appStore.config?.llm_profiles.find(
    (p) => p.name === session.value?.llm_profile,
  );
  return [...new Set(["auto", ...(profile?.reasoning_efforts ?? [])])];
});

function handleRouteChange(id: string): void {
  const seq = ++openSeq;
  isOpening.value = true;
  openError.value = "";
  sendError.value = "";

  // Only trust an already-loaded store when BOTH the sessionId and the loaded
  // session.id match the requested id. A matching sessionId alone is not enough:
  // after a failed open the store may still point at a previous session's id
  // while carrying stale data, and we must not show that old session (A) for B.
  const current = sessionStore.session;
  if (sessionStore.sessionId === id && current?.id === id) {
    isOpening.value = false;
    return;
  }

  sessionStore
    .open(id)
    .catch((cause) => {
      if (seq !== openSeq) return; // superseded by a newer route
      openError.value = cause instanceof Error ? cause.message : String(cause);
    })
    .finally(() => {
      // Even on success, verify the opened session really matches the requested
      // id before hiding the loading state (guards against a stale session).
      if (seq !== openSeq) return;
      if (sessionStore.sessionId !== id || sessionStore.session?.id !== id) {
        if (!openError.value) openError.value = "Не удалось открыть сессию.";
      }
      isOpening.value = false;
    });
}

watch(
  () => props.id,
  (id) => {
    if (id) handleRouteChange(id);
  },
  { immediate: true },
);

async function onSend(payload: SendPayload): Promise<void> {
  // Guard against a double POST: keep the composer busy for the whole call.
  if (isSending.value) return;
  isSending.value = true;
  sendError.value = "";
  // Composer clears its own text + draft on emit, so the payload must be saved
  // here before the HTTP call to restore it if the request fails.
  const pendingText = payload.text;
  try {
    await sessionStore.send(
      pendingText,
      !external.value && payload.effort !== "auto" ? payload.effort : undefined,
    );
    clearDraft();
  } catch (cause) {
    sendError.value = cause instanceof Error ? cause.message : String(cause);
    // Restore the lost text into the composer via a fresh draft key (remount).
    restoreDraft(pendingText);
  } finally {
    isSending.value = false;
  }
}

async function onStop(): Promise<void> {
  try {
    await sessionStore.stop();
  } catch (cause) {
    sendError.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function onApprove(approvalId: string): Promise<void> {
  try {
    await sessionStore.resolve(approvalId, "approve");
  } catch (cause) {
    sendError.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function onReject(payload: RejectPayload): Promise<void> {
  try {
    await sessionStore.resolve(payload.approvalId, "reject", payload.comment || undefined);
  } catch (cause) {
    sendError.value = cause instanceof Error ? cause.message : String(cause);
  }
}

function onOpenChild(id: string): void {
  void router.push(`/s/${id}`);
}
</script>

<template>
  <div class="session-view">
    <template v-if="!isOpening && session && session.id === props.id">
      <header class="session-header">
        <div class="session-heading">
        <h1 class="session-title" :title="session.title" data-testid="session-title">{{ session.title }}</h1>
        <AgentSettings
          controls-id="session-agent-controls"
          :model-label="modelSummary"
          :role-label="modeLabel"
          :effort="selectedEffort"
          :external="external"
        />
        </div>
        <div class="session-meta">
          <span class="session-status" :data-status="session.status" data-testid="session-status">{{ sessionStatusLabel(session.status) }}</span>
        </div>
        <p v-if="external" data-testid="engine-health">{{ engineHealthLabel }}</p>
        <p v-if="external && engineUsage" data-testid="engine-usage">
          Вход: {{ engineUsage.input_tokens ?? "неизвестно" }} ·
          Выход: {{ engineUsage.output_tokens ?? "неизвестно" }} ·
          Кэш: {{ engineUsage.cache_tokens ?? "неизвестно" }} ·
          Стоимость: {{ engineUsage.reported_cost ?? "неизвестна" }}{{ engineUsage.estimated_cost && engineUsage.reported_cost != null ? " (оценка)" : "" }}
        </p>
        <p v-if="session.error" class="session-error" data-testid="session-error">
          {{ session.error }}
        </p>
      </header>

      <ChatTimeline
        v-if="timelineItems.length > 0 || sessionStore.streaming"
        :items="timelineItems"
        data-testid="chat-timeline"
        @approve="onApprove"
        @reject="onReject"
      />
      <p v-else class="session-empty" data-testid="session-empty">
        Сообщений пока нет — начните диалог.
      </p>

      <p v-if="sendError" class="session-error" data-testid="send-error">{{ sendError }}</p>

      <Composer
        :busy="busy"
        :draft-key="draftKey"
        :efforts="efforts"
        controls-target="#session-agent-controls"
        placeholder="Продолжить диалог…"
        @send="onSend"
        @stop="onStop"
        @update:effort="selectedEffort = $event"
      >
        <template #options>
          <div class="agent-control" title="Модель задаётся при создании сессии">
            <span>{{ external ? "Внешний агент" : "Модель" }}</span>
            <span class="agent-value" data-testid="session-profile">{{ modelLabel }}</span>
          </div>
          <div class="agent-control" title="Роль задаётся при создании сессии">
            <span>Роль</span>
            <span class="agent-value" data-testid="session-mode">{{ modeLabel }}</span>
          </div>
        </template>
      </Composer>
    </template>

    <p v-else-if="isOpening" class="session-loading" data-testid="session-loading">
      Загрузка сессии…
    </p>

    <p v-else-if="openError" class="session-error session-error--open" data-testid="open-error">
      {{ openError }}
      <button
        type="button"
        class="open-retry"
        data-testid="open-retry"
        @click="handleRouteChange(props.id)"
      >
        Повторить
      </button>
    </p>
  </div>
</template>

<style scoped>
.session-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.session-header {
  padding: 12px 16px 4px;
  border-bottom: 1px solid var(--border);
}

.session-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 16px;
}

.session-title {
  flex: 1 1 12rem;
  max-width: 100%;
  font-size: 16px;
  font-weight: 650;
  margin: 0;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.session-meta {
  display: flex;
  gap: 8px;
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-muted);
}

.session-status {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--bg-subtle);
  color: var(--text);
}

.session-status::before {
  content: "";
  flex: none;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--session-status-color, var(--text-muted));
}

.session-status[data-status="running"] {
  --session-status-color: var(--info);
}

.session-status[data-status="awaiting_confirmation"] {
  --session-status-color: var(--warning);
}

.session-status[data-status="completed"] {
  --session-status-color: var(--success);
}

.session-status[data-status="error"] {
  --session-status-color: var(--danger);
}

.session-error {
  margin: 8px 16px;
  padding: 8px 12px;
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  color: var(--danger);
  font-size: 13px;
}

.session-error--open {
  display: flex;
  align-items: center;
  gap: 10px;
}

.open-retry {
  padding: 3px 10px;
  border: 1px solid currentColor;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--danger);
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}

.open-retry:hover {
  background: color-mix(in srgb, var(--danger) 12%, transparent);
}

.session-loading,
.session-empty {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 14px;
}

.agent-value {
  min-width: 0;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text);
}
</style>

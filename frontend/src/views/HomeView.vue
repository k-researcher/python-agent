<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useAppStore } from "../stores/app";
import { newSessionProfile } from "../router";
import Composer from "../components/chat/Composer.vue";
import AgentSettings from "../components/chat/AgentSettings.vue";
import EmptyState from "../components/chat/EmptyState.vue";
import type { SendPayload } from "../components/chat/types";
import { toModelOptions } from "../uiAdapters";

const appStore = useAppStore();
const router = useRouter();

const promptDraftKey = ref("draft:new-session");
let draftCounter = 0;

const selectedProfileId = computed({
  get: () => newSessionProfile.value ?? "",
  set: (value: string) => { newSessionProfile.value = value || null; },
});
const mode = ref<"dev" | "ask">("dev");
const selectedEffort = ref("auto");
const isCreating = ref(false);
const error = ref("");

let lifecycleGeneration = 0;

watch(() => appStore.selectedProjectId, () => {
  lifecycleGeneration += 1;
  isCreating.value = false;
  error.value = "";
}, { flush: "sync" });

onBeforeUnmount(() => {
  lifecycleGeneration += 1;
});

const configuredModels = computed(() => toModelOptions(appStore.config));
const defaultProfile = computed(() => appStore.config?.default_llm_profile ?? null);

// If no model was chosen and a default exists, fall back to it.
const effectiveProfileId = computed(() => {
  if (selectedProfileId.value) return selectedProfileId.value;
  return defaultProfile.value ?? "";
});

const defaultModelLabel = computed(() => {
  const profile = configuredModels.value.find((model) => model.id === defaultProfile.value);
  return profile ? `По умолчанию · ${profile.label}` : "По умолчанию";
});
const modelSummary = computed(() => {
  const profile = appStore.config?.llm_profiles.find((model) => model.name === effectiveProfileId.value && model.configured);
  return profile?.model ?? "По умолчанию";
});
const efforts = computed(() => {
  const profile = appStore.config?.llm_profiles.find((model) => model.name === effectiveProfileId.value);
  return [...new Set(["auto", ...(profile?.reasoning_efforts ?? [])])];
});

const hasProjects = computed(() => appStore.projects.length > 0);

function onPick(template: string): void {
  draftCounter += 1;
  const key = `draft:new-session:${draftCounter}`;
  try {
    window.localStorage.setItem(key, template);
  } catch {
    // storage unavailable — the composer just starts empty
  }
  promptDraftKey.value = key;
}

function pickRestoredDraftKey(text: string): void {
  draftCounter += 1;
  const key = `draft:new-session:r${draftCounter}`;
  try {
    window.localStorage.setItem(key, text);
  } catch {
    // storage unavailable — the composer just starts empty
  }
  promptDraftKey.value = key;
}

function clearDraft(): void {
  try {
    window.localStorage.removeItem(promptDraftKey.value);
  } catch {
    // ignore storage write failures
  }
}

async function onCreateSession(payload: SendPayload): Promise<void> {
  if (isCreating.value) return;
  if (!appStore.selectedProjectId) {
    error.value = "Выберите проект в боковой панели.";
    return;
  }
  const generation = lifecycleGeneration;
  const projectId = appStore.selectedProjectId;
  isCreating.value = true;
  error.value = "";
  // Composer clears its own text + draft on emit, so the payload must be saved
  // here before the HTTP call to restore it if the request fails.
  const pendingText = payload.text;
  try {
    const created = await appStore.createSession({
      project_id: projectId,
      prompt: pendingText,
      mode: mode.value,
      llm_profile: effectiveProfileId.value || undefined,
      reasoning_effort: payload.effort !== "auto" ? payload.effort : undefined,
      auto_start: true,
    });
    if (generation !== lifecycleGeneration) return;
    clearDraft();
    await router.push(`/s/${created.id}`);
  } catch (cause) {
    if (generation !== lifecycleGeneration) return;
    error.value = cause instanceof Error ? cause.message : String(cause);
    // Restore the lost text into the composer via a fresh draft key (remount).
    pickRestoredDraftKey(pendingText);
  } finally {
    if (generation === lifecycleGeneration) isCreating.value = false;
  }
}
</script>

<template>
  <div class="home-view">
    <header class="home-header">
      <h1>Новая сессия</h1>
      <AgentSettings
        controls-id="home-agent-controls"
        :model-label="modelSummary"
        :role-label="mode === 'ask' ? 'Консультант' : 'Разработчик'"
        :effort="selectedEffort"
      />
    </header>
    <div class="home-content">
      <EmptyState v-if="!isCreating" @pick="onPick" />
    </div>

    <p v-if="!hasProjects" class="home-error" data-testid="no-projects">
      Нет проектов. Создайте проект в боковой панели.
    </p>

    <p v-if="error" class="home-error" data-testid="home-error">{{ error }}</p>

    <Composer
      :busy="isCreating"
      :stoppable="false"
      :draft-key="promptDraftKey"
      :efforts="efforts"
      controls-target="#home-agent-controls"
      placeholder="Опишите задачу для нового агента…"
      @send="onCreateSession"
      @update:effort="selectedEffort = $event"
    >
      <template #options>
      <label class="agent-control">
        <span>Модель</span>
        <select
          v-model="selectedProfileId"
          data-testid="model-select"
          :disabled="isCreating || configuredModels.length === 0"
        >
          <option value="">
            {{ defaultModelLabel }}
          </option>
          <option v-for="model in configuredModels" :key="model.id" :value="model.id">
            {{ model.label }}
          </option>
        </select>
      </label>
      <label class="agent-control">
        <span>Роль</span>
        <select v-model="mode" :disabled="isCreating" data-testid="mode-select">
          <option value="dev">Разработчик</option>
          <option value="ask">Консультант</option>
        </select>
      </label>
      </template>
    </Composer>
  </div>
</template>

<style scoped>
.home-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.home-error {
  margin: 8px 16px;
  padding: 8px 12px;
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  color: var(--danger);
  font-size: 13px;
}

.home-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 16px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}

.home-header h1 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
}

.home-content {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: grid;
  align-items: center;
}
</style>

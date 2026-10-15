<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { ChevronDown, SlidersHorizontal } from "lucide-vue-next";
import { reasoningLabel } from "./reasoning";

const props = defineProps<{
  controlsId: string;
  modelLabel: string;
  roleLabel: string;
  effort: string;
}>();

const open = ref(false);
const container = ref<HTMLElement | null>(null);
const button = ref<HTMLButtonElement | null>(null);
const panel = ref<HTMLElement | null>(null);
const summary = computed(() => `${props.modelLabel} · ${props.roleLabel} · ${reasoningLabel(props.effort)}`);

async function toggle(): Promise<void> {
  open.value = !open.value;
  if (open.value) {
    await nextTick();
    panel.value?.querySelector<HTMLSelectElement>("select:not(:disabled)")?.focus();
  }
}

function closeOutside(event: PointerEvent): void {
  if (!container.value?.contains(event.target as Node)) open.value = false;
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== "Escape" || !open.value) return;
  event.preventDefault();
  open.value = false;
  button.value?.focus();
}

function onFocusOut(event: FocusEvent): void {
  if (event.relatedTarget && !container.value?.contains(event.relatedTarget as Node)) {
    open.value = false;
  } else if (!event.relatedTarget) {
    void nextTick(() => {
      if (!container.value?.contains(document.activeElement)) open.value = false;
    });
  }
}

onMounted(() => {
  document.addEventListener("pointerdown", closeOutside);
  document.addEventListener("keydown", onKeydown);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", closeOutside);
  document.removeEventListener("keydown", onKeydown);
});
</script>

<template>
  <div ref="container" class="agent-settings" @focusout="onFocusOut">
    <button
      ref="button"
      type="button"
      class="settings-summary"
      aria-label="Параметры агента"
      :aria-controls="controlsId"
      :aria-expanded="open"
      :title="summary"
      data-testid="agent-settings-button"
      @click="toggle"
    >
      <SlidersHorizontal :size="15" aria-hidden="true" />
      <span class="summary-text">{{ summary }}</span>
      <ChevronDown :size="14" aria-hidden="true" />
    </button>
    <div
      :id="controlsId"
      ref="panel"
      :hidden="!open"
      class="settings-panel"
      role="region"
      aria-label="Настройки агента"
      data-testid="agent-settings-panel"
    />
  </div>
</template>

<style scoped>
.agent-settings {
  position: relative;
  min-width: 0;
  max-width: 100%;
}

.settings-summary {
  display: flex;
  align-items: center;
  gap: 8px;
  max-width: 100%;
  min-height: 34px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}

.settings-summary:hover,
.settings-summary[aria-expanded="true"] {
  background: var(--bg-subtle);
  color: var(--text);
}

.settings-summary:focus-visible {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}

.settings-summary svg {
  flex-shrink: 0;
}

.summary-text {
  min-width: 0;
  max-width: 420px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.settings-panel {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  z-index: 40;
  width: min(340px, calc(100vw - 40px));
  max-height: 65vh;
  overflow-y: auto;
  box-sizing: border-box;
  padding: 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow);
}

.agent-settings .settings-panel :deep(.controls) {
  display: grid;
  gap: 14px;
  margin: 0;
}

.agent-settings .settings-panel :deep(.agent-control) {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 6px;
  min-width: 0;
  padding: 0;
  border: 0;
  background: transparent;
  font-size: 12px;
}

.agent-settings .settings-panel :deep(.agent-control select) {
  width: 100%;
  max-width: none;
  min-width: 0;
  height: 36px;
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
}

.agent-settings .settings-panel :deep(.agent-control:focus-within) {
  outline: none;
}

.agent-settings .settings-panel :deep(select:focus-visible) {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}

@media (max-width: 480px) {
  .agent-settings,
  .settings-summary {
    width: 100%;
  }
}
</style>

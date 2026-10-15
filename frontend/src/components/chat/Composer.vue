<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import type { SendPayload } from "./types";
import { reasoningLabel } from "./reasoning";
import { draftGeneration, readDraft, writeDraft } from "../../draftStorage";

const props = withDefaults(
  defineProps<{
    busy?: boolean;
    stoppable?: boolean;
    draftKey?: string;
    efforts?: string[];
    placeholder?: string;
    controlsTarget?: string;
  }>(),
  {
    busy: false,
    stoppable: true,
    draftKey: "",
    efforts: () => ["auto"],
    placeholder: "Что делаем?",
    controlsTarget: "",
  },
);

const emit = defineEmits<{
  (e: "send", payload: SendPayload): void;
  (e: "stop"): void;
  (e: "update:effort", value: string): void;
}>();

const MAX_ROWS = 10;
const accessGeneration = draftGeneration();

const text = ref("");
const effort = ref(props.efforts[0] ?? "auto");
const textarea = ref<HTMLTextAreaElement | null>(null);
const controlsHost = ref<HTMLElement | null>(null);
watch(effort, (value) => emit("update:effort", value), { immediate: true });

function resolveControlsHost(): void {
  controlsHost.value = props.controlsTarget ? document.querySelector<HTMLElement>(props.controlsTarget) : null;
}

onMounted(resolveControlsHost);
watch(() => props.controlsTarget, resolveControlsHost, { flush: "post" });

const canSend = computed(() => text.value.trim().length > 0);

watch(
  () => props.draftKey,
  (key) => {
    text.value = readDraft(key);
    if (!props.efforts.includes(effort.value)) {
      effort.value = props.efforts[0] ?? "auto";
    }
    void nextTick(resize);
  },
  { immediate: true },
);

watch(
  text,
  (value) => {
    writeDraft(props.draftKey, value, accessGeneration);
  },
  { flush: "post" },
);

watch(
  () => props.efforts,
  (levels) => {
    if (!levels.includes(effort.value)) {
      effort.value = levels[0] ?? "auto";
    }
  },
);

function resize(): void {
  const el = textarea.value;
  if (!el) return;
  el.style.height = "auto";
  const lineHeight = Number.parseFloat(window.getComputedStyle(el).lineHeight) || 20;
  const maxHeight = lineHeight * MAX_ROWS;
  el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

function send(): void {
  const content = text.value.trim();
  if (!content || props.busy) return;
  emit("send", { text: content, effort: effort.value });
  text.value = "";
  void nextTick(resize);
}
</script>

<template>
  <div class="composer" :aria-busy="busy" data-testid="composer">
    <div class="input-container">
      <div class="input-row">
        <textarea
          ref="textarea"
          v-model="text"
          rows="1"
          :placeholder="placeholder"
          :disabled="busy"
          data-testid="composer-input"
          @input="resize"
          @keydown="onKeydown"
        />
        <div class="actions">
          <button
            v-if="busy && stoppable"
            type="button"
            class="stop"
            data-testid="composer-stop"
            @click="emit('stop')"
          >
            Стоп
          </button>
          <button
            v-else
            type="button"
            class="send"
            :disabled="busy || !canSend"
            data-testid="composer-send"
            @click="send"
          >
            {{ busy ? "Подождите…" : "Отправить" }}
          </button>
        </div>
      </div>
    </div>
    <Teleport :to="controlsHost ?? 'body'" :disabled="!controlsHost">
    <div class="controls" :class="{ 'controls--hosted': controlsHost }" role="group" aria-label="Параметры агента" data-testid="agent-controls">
      <slot name="options" />
      <label class="agent-control effort">
        <span>Рассуждения</span>
        <select v-model="effort" :disabled="busy" aria-label="Уровень рассуждений" data-testid="composer-effort">
          <option v-for="level in efforts" :key="level" :value="level">{{ reasoningLabel(level) }}</option>
        </select>
      </label>
    </div>
    </Teleport>
  </div>
</template>

<style scoped>
.composer {
  display: block;
  flex-shrink: 0;
  padding: 0.6rem 0.7rem 0.55rem;
  border-top: 1px solid var(--border, #e6e6e9);
  background: var(--surface, #fff);
}

.input-container {
  width: 100%;
}

.input-row {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
}

.actions {
  display: flex;
  flex-shrink: 0;
}

textarea {
  flex: 1;
  min-width: 0;
  min-height: 38px;
  max-height: 200px;
  resize: none;
  box-sizing: border-box;
  padding: 0.5rem 0.65rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  font: inherit;
  font-size: 0.88rem;
  line-height: 1.4;
  background: var(--bg, #f7f7f8);
}

textarea:focus {
  outline: none;
  border-color: var(--link, #2563eb);
}

textarea:disabled {
  opacity: 0.6;
}

button {
  height: 38px;
  padding: 0 1rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  font-size: 0.85rem;
  cursor: pointer;
  white-space: nowrap;
}

button.send {
  background: var(--accent, #111114);
  color: var(--accent-contrast, #fff);
  border-color: var(--accent, #111114);
}

button.send:disabled {
  opacity: 0.45;
  cursor: default;
}

button.stop {
  background: var(--danger, #dc2626);
  color: #fff;
  border-color: var(--danger, #dc2626);
}

.controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}

.controls--hosted {
  margin-top: 0;
}

.controls :deep(.agent-control) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  max-width: 100%;
  font-size: 0.75rem;
  color: var(--text-muted, #6b6b76);
  min-height: 32px;
  padding: 0 8px;
  box-sizing: border-box;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
}

.controls :deep(.agent-control > span:first-child) {
  flex-shrink: 0;
}

.controls :deep(.agent-control select) {
  min-width: 0;
  max-width: 220px;
  height: 30px;
  padding: 3px 6px;
  border: 0;
  border-radius: var(--radius-sm, 6px);
  background: transparent;
  color: var(--text);
  font-size: 0.78rem;
}

.controls :deep(.agent-control:focus-within) {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}
</style>

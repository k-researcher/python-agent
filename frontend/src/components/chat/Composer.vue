<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import type { AttachedFile, SendPayload } from "./types";
import { reasoningLabel } from "./reasoning";
import { draftGeneration, readDraft, writeDraft } from "../../draftStorage";
import { Mic, Paperclip, X } from "lucide-vue-next";
import { addFiles, formatSize } from "./attachments";
import { transcribe } from "../../api/speech";
import { insertText, useVoiceInput } from "./voice";

const props = withDefaults(
  defineProps<{
    busy?: boolean;
    disabled?: boolean;
    stoppable?: boolean;
    draftKey?: string;
    efforts?: string[];
    placeholder?: string;
    controlsTarget?: string;
  }>(),
  {
    busy: false,
    disabled: false,
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

const files = ref<AttachedFile[]>([]);
const fileError = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
const dragging = ref(false);

const canSend = computed(() => text.value.trim().length > 0 || files.value.length > 0);

async function attach(list: Iterable<File>): Promise<void> {
  const result = await addFiles(files.value, list);
  files.value = result.files;
  fileError.value = result.errors.join(" ");
}

function onFileChosen(event: Event): void {
  const input = event.target as HTMLInputElement;
  void attach(Array.from(input.files ?? []));
  input.value = "";
}

function onPaste(event: ClipboardEvent): void {
  const pasted = Array.from(event.clipboardData?.files ?? []);
  if (!pasted.length) return;
  event.preventDefault();
  void attach(pasted);
}

function onDrop(event: DragEvent): void {
  dragging.value = false;
  if (props.busy || props.disabled) return;
  void attach(Array.from(event.dataTransfer?.files ?? []));
}

function removeFile(index: number): void {
  files.value = files.value.filter((_, position) => position !== index);
}

/** Put the files back after a failed send; the text comes back through the draft. */
function restoreFiles(restored: AttachedFile[]): void {
  files.value = restored;
}

defineExpose({ restoreFiles });

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

// Recognized speech goes to the cursor for editing; it is not sent by itself.
const voice = useVoiceInput({
  transcribe: async (audio) => (await transcribe(audio)).text,
  onText: (piece) => {
    const el = textarea.value;
    const start = el?.selectionStart ?? text.value.length;
    const end = el?.selectionEnd ?? text.value.length;
    const result = insertText(text.value, start, end, piece);
    text.value = result.text;
    void nextTick(() => {
      resize();
      el?.focus();
      el?.setSelectionRange(result.caret, result.caret);
    });
  },
});
const voiceState = voice.state;
const voiceError = voice.error;
const recordTime = computed(() => {
  const value = voice.seconds.value;
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
});

function send(): void {
  const content = text.value.trim();
  if ((!content && !files.value.length) || props.busy || props.disabled) return;
  emit("send", { text: content, effort: effort.value, files: files.value });
  text.value = "";
  files.value = [];
  fileError.value = "";
  void nextTick(resize);
}
</script>

<template>
  <div
    class="composer"
    :class="{ 'composer--drop': dragging }"
    :aria-busy="busy"
    data-testid="composer"
    @dragover.prevent="dragging = !busy && !disabled"
    @dragleave="dragging = false"
    @drop.prevent="onDrop"
  >
    <div class="input-container">
      <ul v-if="files.length" class="files" aria-label="Приложенные файлы" data-testid="composer-files">
        <li v-for="(file, index) in files" :key="`${index}:${file.name}`" class="file">
          <span class="file__name" :title="file.name">{{ file.name }}</span>
          <span class="file__size">{{ formatSize(file.size) }}</span>
          <button
            type="button"
            class="file__remove"
            :aria-label="`Убрать файл ${file.name}`"
            :disabled="busy"
            @click="removeFile(index)"
          >
            <X :size="12" aria-hidden="true" />
          </button>
        </li>
      </ul>
      <div class="input-row">
        <textarea
          ref="textarea"
          v-model="text"
          rows="1"
          :placeholder="placeholder"
          :disabled="busy || disabled"
          data-testid="composer-input"
          @input="resize"
          @keydown="onKeydown"
          @paste="onPaste"
        />
        <div class="actions">
          <template v-if="voiceState === 'recording' || voiceState === 'transcribing'">
            <span class="voice-status" role="status" data-testid="voice-status">
              <span v-if="voiceState === 'recording'" class="voice-dot" aria-hidden="true" />
              {{ voiceState === "recording" ? `Запись ${recordTime}` : "Распознаю…" }}
            </span>
            <button type="button" class="voice-cancel" data-testid="voice-cancel" @click="voice.cancel()">Отмена</button>
            <button
              v-if="voiceState === 'recording'"
              type="button"
              class="send"
              data-testid="voice-finish"
              @click="voice.finish()"
            >
              Готово
            </button>
          </template>
          <template v-else>
          <input
            ref="fileInput"
            type="file"
            multiple
            hidden
            data-testid="composer-file-input"
            @change="onFileChosen"
          />
          <button
            type="button"
            class="voice"
            :disabled="busy || disabled"
            aria-label="Приложить текстовый файл"
            title="Приложить текстовый файл"
            data-testid="composer-attach"
            @click="fileInput?.click()"
          >
            <Paperclip :size="16" aria-hidden="true" />
          </button>
          <button
            type="button"
            class="voice"
            :disabled="busy || disabled || voiceState === 'requesting'"
            aria-label="Голосовой ввод"
            title="Голосовой ввод"
            data-testid="voice-start"
            @click="voice.start()"
          >
            <Mic :size="16" aria-hidden="true" />
          </button>
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
            :disabled="busy || disabled || !canSend"
            data-testid="composer-send"
            @click="send"
          >
            {{ busy ? "Подождите…" : "Отправить" }}
          </button>
          </template>
        </div>
      </div>
      <p v-if="fileError" class="voice-error" role="alert" data-testid="composer-file-error">
        {{ fileError }}
        <button type="button" class="voice-error__close" aria-label="Скрыть" @click="fileError = ''">×</button>
      </p>
      <p v-if="voiceError" class="voice-error" role="alert" data-testid="voice-error">
        {{ voiceError }}
        <button type="button" class="voice-error__close" aria-label="Скрыть" @click="voiceError = ''">×</button>
      </p>
    </div>
    <Teleport :to="controlsHost ?? 'body'" :disabled="!controlsHost">
    <div class="controls" :class="{ 'controls--hosted': controlsHost }" role="group" aria-label="Параметры агента" data-testid="agent-controls">
      <slot name="options" />
      <label v-if="efforts.length" class="agent-control effort">
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

.composer--drop {
  outline: 2px dashed var(--link);
  outline-offset: -4px;
}

.files {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0 0 6px;
  padding: 0;
  list-style: none;
}

.file {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  min-width: 0;
  height: 26px;
  padding: 0 4px 0 8px;
  box-sizing: border-box;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  font-size: 0.75rem;
}

.file__name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file__size {
  flex-shrink: 0;
  color: var(--text-muted);
}

button.file__remove {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--text-muted);
}

.input-row {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
}

.actions {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  gap: 0.4rem;
}

button.voice {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  padding: 0;
  background: var(--bg-subtle);
  color: var(--text);
}

button.voice:disabled {
  opacity: 0.45;
  cursor: default;
}

button.voice-cancel {
  background: transparent;
  color: var(--text);
}

.voice-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.82rem;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.voice-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--danger);
  animation: voice-pulse 1.2s ease-in-out infinite;
}

@keyframes voice-pulse {
  50% { opacity: 0.35; }
}

@media (prefers-reduced-motion: reduce) {
  .voice-dot { animation: none; }
}

.voice-error {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 6px 0 0;
  font-size: 0.8rem;
  color: var(--danger);
}

.voice-error__close {
  height: auto;
  margin-left: auto;
  padding: 0 4px;
  border: 0;
  background: transparent;
  color: inherit;
  font-size: 1rem;
  line-height: 1;
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
  color: var(--accent-contrast, #fff);
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

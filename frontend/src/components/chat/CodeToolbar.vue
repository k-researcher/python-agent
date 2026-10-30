<script setup lang="ts">
// Buttons inside the top right corner of a code block: copy the code and switch line wrap.
// The block owns the wrap state; this component only reports the click.

import { onBeforeUnmount, ref } from "vue";
import { Check, Copy, TextWrap } from "lucide-vue-next";

const props = defineProps<{ text: () => string; wrapped: boolean }>();
const emit = defineEmits<{ toggleWrap: [] }>();

const copyState = ref<"idle" | "done" | "failed">("idle");
let timer: number | undefined;

async function copy(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.text());
    copyState.value = "done";
  } catch {
    copyState.value = "failed";
  }
  window.clearTimeout(timer);
  timer = window.setTimeout(() => {
    copyState.value = "idle";
  }, 1200);
}

onBeforeUnmount(() => window.clearTimeout(timer));

const COPY_LABELS = { idle: "Копировать", done: "Скопировано", failed: "Не удалось скопировать" };
</script>

<template>
  <div class="code-toolbar" data-testid="code-toolbar">
    <button
      type="button"
      class="code-toolbar__btn wrap-btn"
      :aria-pressed="wrapped"
      :title="wrapped ? 'Без переноса строк' : 'Переносить строки'"
      :aria-label="wrapped ? 'Без переноса строк' : 'Переносить строки'"
      data-testid="code-wrap"
      @click="emit('toggleWrap')"
    >
      <TextWrap :size="14" aria-hidden="true" />
    </button>
    <button
      type="button"
      class="code-toolbar__btn copy-btn"
      :data-state="copyState"
      :title="COPY_LABELS[copyState]"
      :aria-label="COPY_LABELS[copyState]"
      data-testid="code-copy"
      @click="copy"
    >
      <Check v-if="copyState === 'done'" :size="14" aria-hidden="true" />
      <Copy v-else :size="14" aria-hidden="true" />
      <span v-if="copyState !== 'idle'" class="code-toolbar__note" role="status">{{ COPY_LABELS[copyState] }}</span>
    </button>
  </div>
</template>

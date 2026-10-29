<script setup lang="ts">
// Collapsible "reasoning" block. Collapsed by default; while streaming it
// shows an animated "Думаю…" hint and can be expanded to follow the live text.

import { computed, ref, watch } from "vue";

const props = withDefaults(defineProps<{ text?: string; streaming?: boolean }>(), {
  text: "",
  streaming: false,
});

const expanded = ref(false);
const body = ref<HTMLElement | null>(null);
// The live text follows the end until the user scrolls up, and again after they scroll back down.
const followEnd = ref(true);

const charCount = computed(() => props.text.length);

const countLabel = computed<string>(() => {
  const count = charCount.value;
  const suffix = count === 1 ? "символ" : count >= 2 && count <= 4 ? "символа" : "символов";
  return `${count} ${suffix}`;
});

function toggle(): void {
  expanded.value = !expanded.value;
}

function onScroll(): void {
  const el = body.value;
  if (el) followEnd.value = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
}

watch(
  () => props.text,
  () => {
    if (props.streaming && followEnd.value && body.value) body.value.scrollTop = body.value.scrollHeight;
  },
  { flush: "post" },
);
</script>

<template>
  <div class="reasoning-block" :data-streaming="streaming">
    <button
      type="button"
      class="reasoning-toggle"
      :aria-expanded="expanded"
      data-testid="reasoning-toggle"
      @click="toggle"
    >
      <span class="caret" :class="{ open: expanded }">▸</span>
      <template v-if="streaming"><span class="thinking-dots">Думаю…</span> · {{ countLabel }}</template>
      <span v-else>Рассуждения · {{ countLabel }}</span>
    </button>
    <div
      v-if="expanded"
      ref="body"
      class="reasoning-body"
      :class="{ live: streaming }"
      data-testid="reasoning-body"
      @scroll="onScroll"
    >
      <pre class="reasoning-text">{{ text }}</pre>
    </div>
  </div>
</template>

<style scoped>
.reasoning-block {
  margin: 0.35rem 0;
}

.reasoning-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.1rem 0.35rem;
  border: 0;
  border-radius: var(--radius-sm, 6px);
  background: transparent;
  color: var(--text-muted, #6b6b76);
  font-size: 0.78rem;
  cursor: pointer;
}
.reasoning-toggle:hover {
  background: var(--bg-subtle, #f4f4f6);
}

.caret {
  display: inline-block;
  transition: transform 0.15s ease;
  font-size: 0.7rem;
}
.caret.open {
  transform: rotate(90deg);
}

.thinking-dots {
  animation: pulse 1.1s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .thinking-dots {
    animation: none;
  }
}

.reasoning-body {
  margin-top: 0.3rem;
  padding: 0.5rem 0.65rem;
  border-left: 2px solid var(--border, #e6e6e9);
  color: var(--text-muted, #6b6b76);
}
.reasoning-body.live {
  max-height: 40vh;
  overflow-y: auto;
}

.reasoning-text {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
  font-size: 0.82rem;
  line-height: 1.5;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 0.35;
  }
  50% {
    opacity: 1;
  }
}
</style>

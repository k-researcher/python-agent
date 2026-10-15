<script setup lang="ts">
// Renders a chat message as sanitized markdown with lazy shiki highlighting
// and a copy button on every code block.

import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { highlightCodeBlocks, renderMarkdown, wrapCodeBlocksInPre } from "./markdown";

const props = withDefaults(defineProps<{ source?: string; dark?: boolean }>(), {
  source: "",
  dark: undefined,
});

const isDark = computed(() => {
  if (props.dark !== undefined) return props.dark;
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  return false;
});

const html = ref("");
const rootEl = ref<HTMLElement | null>(null);
let generation = 0;
let disposed = false;

onBeforeUnmount(() => {
  disposed = true;
});

watch(
  [() => props.source, isDark],
  async () => {
    const gen = ++generation;
    const base = wrapCodeBlocksInPre(renderMarkdown(props.source ?? ""));
    if (disposed || gen !== generation) return;
    // First paint plain code blocks and attach copy buttons before async
    // shiki highlighting finishes, so copying never depends on it.
    html.value = base;
    await nextTick();
    if (disposed || gen !== generation) return;
    attachCopyButtons();
    let highlighted = base;
    try {
      highlighted = await highlightCodeBlocks(base, isDark.value);
    } catch {
      // Shiki unavailable -> keep the plain, wrapped blocks.
      highlighted = base;
    }
    if (disposed || gen !== generation) return;
    html.value = highlighted;
    await nextTick();
    if (disposed || gen !== generation) return;
    attachCopyButtons();
  },
  { immediate: true, flush: "post" },
);

async function copyCode(button: HTMLButtonElement, pre: HTMLElement): Promise<void> {
  const code = pre.querySelector("code");
  const text = code?.textContent ?? "";
  const original = button.textContent ?? "";
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Скопировано";
  } catch {
    button.textContent = "Не удалось скопировать";
  }
  window.setTimeout(() => {
    button.textContent = original;
    button.disabled = false;
  }, 1200);
}

function attachCopyButtons(): void {
  const root = rootEl.value;
  if (!root) return;
  // Idempotency marker: skip <pre> blocks that already carry a copy button so
  // a re-run after source/highlight updates never creates a second one.
  for (const pre of Array.from(root.querySelectorAll<HTMLElement>("pre"))) {
    if (pre.dataset.hasCopyBtn === "true") continue;
    if (pre.parentElement?.classList.contains("code-block")) continue;
    const block = document.createElement("div");
    block.className = "code-block";
    const toolbar = document.createElement("div");
    toolbar.className = "code-block__toolbar";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.textContent = "Копировать";
    button.addEventListener("click", () => {
      button.disabled = true;
      void copyCode(button, pre);
    });
    pre.dataset.hasCopyBtn = "true";
    pre.parentNode?.insertBefore(block, pre);
    block.appendChild(pre);
    toolbar.appendChild(button);
    block.insertBefore(toolbar, pre);
  }
}
</script>

<template>
  <div ref="rootEl" class="markdown-body" v-html="html" data-testid="markdown-body" />
</template>

<style scoped>
.markdown-body {
  line-height: 1.55;
  overflow-wrap: anywhere;
  color: var(--text, #111114);
}

.markdown-body :deep(p) {
  margin: 0.4rem 0;
}
.markdown-body :deep(p:first-child) {
  margin-top: 0;
}
.markdown-body :deep(p:last-child) {
  margin-bottom: 0;
}

.markdown-body :deep(a) {
  color: var(--link, #2563eb);
  text-decoration: underline;
  text-underline-offset: 2px;
}
</style>

<!-- Code blocks and their copy buttons are injected via v-html at runtime, so
     they can't carry the scoped-data attribute; style them globally. -->
<style>
.code-block {
  position: relative;
  margin: 0.6rem 0;
}

.code-block__toolbar {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-bottom: 0.2rem;
}

.code-block pre {
  margin: 0;
  padding: 1rem 0.9rem 0.8rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  overflow-x: auto;
  background: var(--bg-subtle, #f4f4f6);
  font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
  font-size: 0.82rem;
  line-height: 1.5;
}

.code-block .copy-btn {
  padding: 0.15rem 0.5rem;
  font-size: 0.7rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  background: var(--surface, #fff);
  color: var(--text-muted, #6b6b76);
  cursor: pointer;
}
.code-block .copy-btn:hover {
  color: var(--text, #111114);
  border-color: var(--accent, #111114);
}
</style>

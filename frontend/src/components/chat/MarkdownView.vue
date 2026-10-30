<script setup lang="ts">
// Renders a chat message as sanitized markdown with lazy shiki highlighting.
// Every code block gets the shared toolbar (copy, line wrap) inside its corner.

import { h, nextTick, onBeforeUnmount, ref, render, watch } from "vue";
import { highlightCodeBlocks, renderMarkdown, wrapCodeBlocksInPre } from "./markdown";
import CodeToolbar from "./CodeToolbar.vue";

const props = withDefaults(defineProps<{ source?: string }>(), { source: "" });

const html = ref("");
const rootEl = ref<HTMLElement | null>(null);
// Toolbar hosts mounted into the v-html content; unmount them before it is replaced.
let hosts: HTMLElement[] = [];
let generation = 0;
let disposed = false;

function unmountToolbars(): void {
  for (const host of hosts) render(null, host);
  hosts = [];
}

async function show(content: string, gen: number): Promise<boolean> {
  unmountToolbars();
  html.value = content;
  await nextTick();
  if (disposed || gen !== generation) return false;
  attachToolbars();
  return true;
}

onBeforeUnmount(() => {
  disposed = true;
  unmountToolbars();
});

watch(
  () => props.source,
  async () => {
    const gen = ++generation;
    const base = wrapCodeBlocksInPre(renderMarkdown(props.source ?? ""));
    // First paint plain code blocks with their toolbars before async shiki
    // highlighting finishes, so copying never depends on it.
    if (!(await show(base, gen))) return;
    let highlighted = base;
    try {
      highlighted = await highlightCodeBlocks(base);
    } catch {
      // Shiki unavailable -> keep the plain, wrapped blocks.
      highlighted = base;
    }
    if (disposed || gen !== generation) return;
    await show(highlighted, gen);
  },
  { immediate: true, flush: "post" },
);

function attachToolbars(): void {
  const root = rootEl.value;
  if (!root) return;
  for (const pre of Array.from(root.querySelectorAll<HTMLElement>("pre"))) {
    // Skip a <pre> that is already inside a block, so a re-run never adds a second toolbar.
    if (pre.parentElement?.classList.contains("code-block")) continue;
    const block = document.createElement("div");
    block.className = "code-block";
    const host = document.createElement("div");
    host.className = "code-block__toolbar";
    pre.parentNode?.insertBefore(block, pre);
    block.appendChild(host);
    block.appendChild(pre);
    hosts.push(host);
    render(
      h({
        setup() {
          const wrapped = ref(true);
          return () =>
            h(CodeToolbar, {
              text: () => pre.querySelector("code")?.textContent ?? "",
              wrapped: wrapped.value,
              onToggleWrap: () => {
                wrapped.value = !wrapped.value;
                block.classList.toggle("code-block--nowrap", !wrapped.value);
              },
            });
        },
      }),
      host,
    );
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

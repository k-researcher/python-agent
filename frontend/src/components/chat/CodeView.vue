<script setup lang="ts">
// Read-only code block for tool cards: dark background, syntax colours when the
// language is known, and optional diff line kinds. Plain text shows at once; the
// colours replace it when the highlighter has loaded. Long lines wrap by default,
// so the block never needs a horizontal scroll unless the user turns wrap off.

import { computed, ref, watch } from "vue";
import type { DiffLineType } from "./diff";
import { diffMarker } from "./diff";
import { loadTokenizer, MAX_HIGHLIGHT_LINES, type CodeToken } from "../../highlight";
import CodeToolbar from "./CodeToolbar.vue";

interface CodeLine {
  text: string;
  type?: DiffLineType;
}

const props = withDefaults(
  defineProps<{ lines: CodeLine[]; lang?: string | null; diff?: boolean; start?: number | null }>(),
  { lang: null, diff: true, start: null },
);

const tokens = ref<CodeToken[][] | null>(null);
const wrapped = ref(true);
const code = computed(() => props.lines.map((line) => line.text).join("\n"));
// Copy gives the code as it is after the change: removed lines stay out.
const copyText = () =>
  props.lines
    .filter((line) => line.type !== "remove")
    .map((line) => line.text)
    .join("\n");
let request = 0;

// Colour the old and the new side of a diff apart, so a construct that spans lines
// (a docstring, for example) is read in the order the file really has.
async function colour(lang: string): Promise<CodeToken[][] | null> {
  const tokenize = await loadTokenizer();
  const result: CodeToken[][] = new Array(props.lines.length);
  for (const side of ["remove", "add"] as const) {
    const indexes = props.lines.flatMap((line, i) => (line.type === (side === "add" ? "remove" : "add") ? [] : [i]));
    const changed = indexes.some((i) => props.lines[i].type === side);
    if (!changed && (side === "remove" || indexes.length === 0)) continue;
    const lines = await tokenize(indexes.map((i) => props.lines[i].text).join("\n"), lang);
    if (lines.length !== indexes.length) return null;
    indexes.forEach((lineIndex, j) => {
      if (side === "add" || props.lines[lineIndex].type === "remove") result[lineIndex] = lines[j];
    });
  }
  return result.length === props.lines.length && result.every(Boolean) ? result : null;
}

watch(
  [code, () => props.lang],
  async () => {
    const current = ++request;
    tokens.value = null;
    if (!props.lang || props.lines.length === 0 || props.lines.length > MAX_HIGHLIGHT_LINES) return;
    try {
      const result = await colour(props.lang);
      // Use the colours only when they still match the lines on screen.
      if (current === request) tokens.value = result;
    } catch {
      // Unknown grammar or load failure: the plain text stays.
    }
  },
  { immediate: true },
);

function marker(line: CodeLine): string {
  const sign = diffMarker(line.type ?? "context");
  return sign === "-" ? "−" : sign.trim();
}
</script>

<template>
  <div class="code-frame">
    <CodeToolbar :text="copyText" :wrapped="wrapped" @toggle-wrap="wrapped = !wrapped" />
    <div class="code-view" :class="{ 'code-view--nowrap': !wrapped }" :data-lang="lang ?? ''" data-testid="code-view">
      <div
        v-for="(line, i) in lines"
        :key="i"
        class="code-line"
        :data-type="line.type ?? 'context'"
        data-testid="code-line"
      ><span v-if="start !== null" class="code-number" data-testid="code-number">{{ start + i }}</span><span
          v-if="diff && line.type && line.type !== 'context'" class="code-marker">{{ marker(line) }}</span><span
          v-else-if="diff"
          class="code-marker"
        /><span class="code-text"><template v-if="tokens"><span
              v-for="(token, j) in tokens[i]"
              :key="j"
              :style="{ color: token.color }"
            >{{ token.content }}</span></template><template v-else>{{ line.text }}</template></span></div>
    </div>
  </div>
</template>

<style scoped>
.code-frame {
  position: relative;
}

.code-view {
  max-height: 28rem;
  overflow: auto;
  padding: 0.45rem 0;
  border-radius: var(--radius-sm, 6px);
  background: var(--code-bg, #2a2420);
  color: var(--code-text, #e3d9cc);
  font-family: var(--font-mono, ui-monospace, Menlo, monospace);
  font-size: 0.78rem;
  line-height: 1.5;
}

.code-line {
  display: flex;
  padding: 0 0.75rem 0 0.4rem;
}
.code-view--nowrap .code-line {
  min-width: max-content;
}
.code-line[data-type="add"] {
  background: rgba(87, 171, 90, 0.16);
}
.code-line[data-type="remove"] {
  background: rgba(229, 83, 75, 0.16);
}

.code-marker {
  flex-shrink: 0;
  width: 1.2rem;
  color: var(--code-muted, #9a8b7d);
  user-select: none;
}
.code-line[data-type="add"] .code-marker {
  color: #57ab5a;
}
.code-line[data-type="remove"] .code-marker {
  color: #e5534b;
}

.code-number {
  flex-shrink: 0;
  min-width: 2.5rem;
  padding-right: 0.75rem;
  text-align: right;
  color: var(--code-muted, #9a8b7d);
  user-select: none;
}

/* A wrapped line continues under its own text, right of the number and the marker. */
.code-text {
  flex: 1;
  min-width: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.code-view--nowrap .code-text {
  white-space: pre;
  overflow-wrap: normal;
}
</style>

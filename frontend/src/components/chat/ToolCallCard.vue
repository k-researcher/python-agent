<script setup lang="ts">
// Card for one tool call in the timeline: name, brief argument, status badge,
// expandable "Кратко" / "JSON" tabs, shell stdout/stderr, and a simple
// old -> new diff for file-editing tools. Forwards approval actions up.

import { computed, ref, type Component } from "vue";
import {
  BookOpen,
  Bot,
  FilePen,
  FileText,
  Globe,
  Search,
  SquareTerminal,
  Trash2,
  Wrench,
} from "lucide-vue-next";
import type { RejectPayload, ToolCallView } from "./types";
import ApprovalInline from "./ApprovalInline.vue";
import CodeView from "./CodeView.vue";
import { languageForOutput, languageForPath } from "../../highlight";
import {
  briefArgument,
  editDiff,
  fileEdits,
  prettyJson,
  resultBlocks,
  shellOutput,
  statusLabel,
} from "./toolDetails";

const props = withDefaults(defineProps<{ call: ToolCallView; hotkeys?: boolean }>(), {
  hotkeys: false,
});

const emit = defineEmits<{
  (e: "approve", approvalId: string): void;
  (e: "reject", payload: RejectPayload): void;
}>();

const open = ref(false);
const tab = ref<"brief" | "json">("brief");

// One icon per kind of work, so a long timeline reads at a glance.
const ICONS: Record<string, [string, Component]> = {
  read_file: ["read", FileText],
  list_dir: ["read", FileText],
  glob: ["search", Search],
  search_files: ["search", Search],
  write_file: ["edit", FilePen],
  edit_file: ["edit", FilePen],
  multi_edit: ["edit", FilePen],
  remove_file: ["remove", Trash2],
  shell: ["shell", SquareTerminal],
  run_agent: ["agent", Bot],
  http_get: ["web", Globe],
  knowledge_search: ["knowledge", BookOpen],
};
const icon = computed(() => ICONS[props.call.name] ?? ["other", Wrench]);

const brief = computed(() => briefArgument(props.call));
const label = computed(() => statusLabel(props.call.status));
const isShell = computed(() => props.call.name === "shell");
const isFileEdit = computed(
  () => props.call.name === "write_file" || props.call.name === "edit_file" || props.call.name === "multi_edit",
);

const json = computed(() => prettyJson(props.call));
const jsonLines = computed(() => textLines(json.value));
const command = computed(() => {
  const value = props.call.arguments?.command;
  return typeof value === "string" ? value : "";
});

function textLines(text: string): { text: string }[] {
  return text.replace(/\n$/, "").split("\n").map((line) => ({ text: line }));
}

// A unified patch: "+" and "-" lines are changes, headers and the rest are context.
function patchLines(patch: string): { text: string; type: "add" | "remove" | "context" }[] {
  return textLines(patch).map(({ text }) => {
    if (text.startsWith("+") && !text.startsWith("+++")) return { text: text.slice(1), type: "add" };
    if (text.startsWith("-") && !text.startsWith("---")) return { text: text.slice(1), type: "remove" };
    return { text, type: "context" };
  });
}
const shell = computed(() => shellOutput(props.call));
const edits = computed(() => fileEdits(props.call));
const blocks = computed(() => resultBlocks(props.call));

function toggle(): void {
  open.value = !open.value;
}

function approveForward(id: string): void {
  emit("approve", id);
}
function rejectForward(payload: RejectPayload): void {
  emit("reject", payload);
}
</script>

<template>
  <div class="tool-call" :data-status="call.status" data-testid="tool-call">
    <div class="tool-head" role="button" tabindex="0" @click="toggle">
      <span class="caret" :class="{ open }">▸</span>
      <component
        :is="icon[1]"
        class="tool-icon"
        :size="15"
        aria-hidden="true"
        :data-kind="icon[0]"
        data-testid="tool-icon"
      />
      <span class="tool-name" data-testid="tool-name">{{ call.name }}</span>
      <span class="tool-arg" :title="brief" data-testid="tool-brief">{{ brief }}</span>
      <span class="status-badge status-pill" :data-status="call.status" data-testid="tool-status">{{ label }}</span>
    </div>

    <ApprovalInline
      v-if="call.approval && call.status === 'awaiting_approval'"
      :approval-id="call.approval.id"
      :risk-level="call.approval.risk_level"
      :hotkeys="hotkeys"
      :scope="call.approval.scope"
      :executor="call.approval.executor"
      @approve="approveForward"
      @reject="rejectForward"
    />

    <div v-if="open" class="tool-detail" data-testid="tool-detail">
      <div class="tabs">
        <button
          type="button"
          class="tab"
          :class="{ active: tab === 'brief' }"
          data-testid="tab-brief"
          @click="tab = 'brief'"
        >
          Кратко
        </button>
        <button
          type="button"
          class="tab"
          :class="{ active: tab === 'json' }"
          data-testid="tab-json"
          @click="tab = 'json'"
        >
          JSON
        </button>
      </div>

      <div v-if="tab === 'json'" class="json-panel" data-testid="json-panel">
        <CodeView :lines="jsonLines" lang="json" :diff="false" />
      </div>

      <div v-else class="brief-panel" data-testid="brief-panel">
        <div v-if="call.result && !call.result.success && call.result.error" class="call-error">
          {{ call.result.error }}
        </div>

        <div v-if="call.name === 'engine.diff' && call.result?.patch != null" class="engine-patch" data-testid="engine-patch">
          <span>{{ call.result.status === "applied" ? "Применено" : "Предложено" }}</span>
          <div v-for="path in call.result.paths" :key="path">{{ path }}</div>
          <CodeView class="diff" :lines="patchLines(call.result.patch)" />
        </div>

        <div v-else-if="isShell" class="shell-result" data-testid="shell-result">
          <div v-if="command" class="output" data-testid="shell-command">
            <CodeView :lines="textLines('$ ' + command)" lang="bash" :diff="false" />
          </div>
          <div v-if="shell.stdout || shell.stdout === ''" class="output" data-testid="shell-stdout">
            <div class="output-label">stdout</div>
            <CodeView
              :lines="textLines(shell.stdout || '—')"
              :lang="languageForOutput(shell.stdout || '')"
              :diff="false"
            />
          </div>
          <div v-if="shell.stderr" class="output err" data-testid="shell-stderr">
            <div class="output-label">stderr</div>
            <CodeView :lines="textLines(shell.stderr)" :lang="languageForOutput(shell.stderr)" :diff="false" />
          </div>
          <div v-if="shell.exitCode !== null" class="exit-code" data-testid="shell-exit">
            exit code: <span :class="shell.exitCode === 0 ? 'ok' : 'err'">{{ shell.exitCode }}</span>
          </div>
        </div>

        <div v-else-if="isFileEdit" class="file-edits" data-testid="file-edits">
          <div v-if="edits.length === 0" class="muted">Нет изменений для показа</div>
          <div v-for="(pair, i) in edits" :key="i" class="edit-block">
            <div v-if="pair.path" class="edit-path" data-testid="edit-path">{{ pair.path }}</div>
            <CodeView class="diff" :lines="editDiff(pair)" :lang="languageForPath(pair.path)" />
          </div>
        </div>

        <div v-else-if="blocks.length" class="result-blocks" data-testid="result-blocks">
          <div v-for="(block, i) in blocks" :key="i" class="output">
            <div v-if="block.label" class="output-label">{{ block.label }}</div>
            <CodeView :lines="block.lines" :lang="block.lang" :diff="false" :start="block.start ?? null" />
          </div>
        </div>

        <div v-else class="muted">{{ brief }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tool-call {
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  background: var(--surface, #fff);
  padding: 0.45rem 0.6rem;
  margin: 0.25rem 0;
}

.tool-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  user-select: none;
}

.caret {
  display: inline-block;
  transition: transform 0.15s ease;
  font-size: 0.7rem;
  color: var(--text-muted, #6b6b76);
}
.caret.open {
  transform: rotate(90deg);
}

.tool-icon {
  flex-shrink: 0;
  color: var(--text-muted, #6b6b76);
}

.tool-name {
  font-family: var(--font-mono, ui-monospace, Menlo, monospace);
  font-size: 0.8rem;
  font-weight: 600;
}

.tool-arg {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-muted, #6b6b76);
  font-size: 0.78rem;
}

.status-badge[data-status="awaiting_approval"] {
  --status-color: var(--warning, #d97706);
}
.status-badge[data-status="running"] {
  --status-color: var(--info, #2563eb);
}
.status-badge[data-status="completed"] {
  --status-color: var(--success, #16a34a);
}
.status-badge[data-status="error"],
.status-badge[data-status="rejected"] {
  --status-color: var(--danger, #dc2626);
}

.tool-detail {
  margin-top: 0.5rem;
  border-top: 1px solid var(--border, #e6e6e9);
  padding-top: 0.45rem;
}

.tabs {
  display: flex;
  gap: 0.35rem;
  margin-bottom: 0.45rem;
}
.tab {
  padding: 0.2rem 0.6rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  background: transparent;
  font-size: 0.75rem;
  cursor: pointer;
  color: var(--text-muted, #6b6b76);
}
.tab.active {
  background: var(--bg-subtle, #f4f4f6);
  color: var(--text, #111114);
}

.output + .output {
  margin-top: 0.4rem;
}
.output-label {
  margin-bottom: 0.2rem;
  color: var(--text-muted, #6b6b76);
  font-family: var(--font-mono, ui-monospace, Menlo, monospace);
  font-size: 0.72rem;
}
.output.err .output-label {
  color: var(--danger, #dc2626);
}

.exit-code {
  margin-top: 0.35rem;
  font-size: 0.75rem;
  font-family: var(--font-mono, ui-monospace, Menlo, monospace);
}
.exit-code .ok {
  color: var(--success, #16a34a);
}
.exit-code .err {
  color: var(--danger, #dc2626);
}

.edit-block {
  margin-bottom: 0.5rem;
}
.edit-path {
  font-size: 0.75rem;
  font-family: var(--font-mono, ui-monospace, Menlo, monospace);
  margin-bottom: 0.2rem;
  color: var(--text, #111114);
}

.call-error {
  color: var(--danger, #dc2626);
  font-size: 0.78rem;
  margin-bottom: 0.35rem;
}
.muted {
  color: var(--text-muted, #6b6b76);
  font-size: 0.78rem;
}
</style>

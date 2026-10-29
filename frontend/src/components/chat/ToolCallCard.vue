<script setup lang="ts">
// Card for one tool call in the timeline: name, brief argument, status badge,
// expandable "Кратко" / "JSON" tabs, shell stdout/stderr, and a simple
// old -> new diff for file-editing tools. Forwards approval actions up.

import { computed, ref } from "vue";
import type { RejectPayload, ToolCallView } from "./types";
import ApprovalInline from "./ApprovalInline.vue";
import {
  briefArgument,
  editDiff,
  fileEdits,
  prettyJson,
  shellOutput,
  statusLabel,
} from "./toolDetails";
import { diffMarker } from "./diff";

const props = withDefaults(defineProps<{ call: ToolCallView; hotkeys?: boolean }>(), {
  hotkeys: false,
});

const emit = defineEmits<{
  (e: "approve", approvalId: string): void;
  (e: "reject", payload: RejectPayload): void;
}>();

const open = ref(false);
const tab = ref<"brief" | "json">("brief");

const brief = computed(() => briefArgument(props.call));
const label = computed(() => statusLabel(props.call.status));
const isShell = computed(() => props.call.name === "shell");
const isFileEdit = computed(
  () => props.call.name === "write_file" || props.call.name === "edit_file" || props.call.name === "multi_edit",
);

const json = computed(() => prettyJson(props.call));
const shell = computed(() => shellOutput(props.call));
const edits = computed(() => fileEdits(props.call));

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
      <span class="tool-icon" aria-hidden="true">⚙</span>
      <span class="tool-name" data-testid="tool-name">{{ call.name }}</span>
      <span class="tool-arg" :title="brief" data-testid="tool-brief">{{ brief }}</span>
      <span class="status-badge" :data-status="call.status" data-testid="tool-status">{{ label }}</span>
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
        <pre>{{ json }}</pre>
      </div>

      <div v-else class="brief-panel" data-testid="brief-panel">
        <div v-if="call.result && !call.result.success && call.result.error" class="call-error">
          {{ call.result.error }}
        </div>

        <div v-if="call.name === 'engine.diff' && call.result?.patch != null" class="engine-patch" data-testid="engine-patch">
          <span>{{ call.result.status === "applied" ? "Применено" : "Предложено" }}</span>
          <div v-for="path in call.result.paths" :key="path">{{ path }}</div>
          <pre class="mono diff">{{ call.result.patch }}</pre>
        </div>

        <div v-else-if="isShell" class="shell-result" data-testid="shell-result">
          <pre v-if="shell.stdout || shell.stdout === ''" class="mono" data-testid="shell-stdout">
stdout
{{ shell.stdout || "—" }}</pre
          >
          <pre v-if="shell.stderr" class="mono err" data-testid="shell-stderr">
stderr
{{ shell.stderr }}</pre
          >
          <div v-if="shell.exitCode !== null" class="exit-code" data-testid="shell-exit">
            exit code: <span :class="shell.exitCode === 0 ? 'ok' : 'err'">{{ shell.exitCode }}</span>
          </div>
        </div>

        <div v-else-if="isFileEdit" class="file-edits" data-testid="file-edits">
          <div v-if="edits.length === 0" class="muted">Нет изменений для показа</div>
          <div v-for="(pair, i) in edits" :key="i" class="edit-block">
            <div v-if="pair.path" class="edit-path" data-testid="edit-path">{{ pair.path }}</div>
            <pre class="mono diff"><template
                v-for="(line, j) in editDiff(pair)"
                :key="j"
              >{{ diffMarker(line.type) }} {{ line.text }}
</template></pre>
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

.status-badge {
  flex-shrink: 0;
  padding: 0.1rem 0.5rem;
  border-radius: 999px;
  background: var(--bg-subtle, #f4f4f6);
  color: var(--text-muted, #6b6b76);
  font-size: 0.7rem;
}
.status-badge[data-status="awaiting_approval"] {
  background: var(--warning, #d97706);
  color: #fff;
}
.status-badge[data-status="running"] {
  background: var(--link, #2563eb);
  color: #fff;
}
.status-badge[data-status="completed"] {
  background: var(--success, #16a34a);
  color: #fff;
}
.status-badge[data-status="error"],
.status-badge[data-status="rejected"] {
  background: var(--danger, #dc2626);
  color: #fff;
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

.json-panel pre,
.shell-result pre,
.edit-block pre,
.engine-patch pre {
  margin: 0;
  padding: 0.45rem 0.55rem;
  background: var(--bg-subtle, #f4f4f6);
  border-radius: var(--radius-sm, 6px);
  overflow-x: auto;
  font-family: var(--font-mono, ui-monospace, Menlo, monospace);
  font-size: 0.75rem;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-all;
}

.shell-result pre.err {
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

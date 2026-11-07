<script setup lang="ts">
// Inline approval controls for a pending tool call: risk level, Approve /
// Reject. Reject reveals an optional comment field. When `hotkeys` is true
// the A/R shortcuts work unless focus sits in a text input.

import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { RejectPayload } from "./types";

const props = withDefaults(
  defineProps<{ approvalId: string; riskLevel: string; hotkeys?: boolean; scope?: "request" | "turn"; executor?: "python" | "engine" }>(),
  { hotkeys: false },
);

const emit = defineEmits<{
  (e: "approve", approvalId: string): void;
  (e: "reject", payload: RejectPayload): void;
}>();

const commentVisible = ref(false);
const comment = ref("");

const riskLabel = computed(() => {
  switch ((props.riskLevel ?? "").toLowerCase()) {
    case "high":
      return "Высокий риск";
    case "medium":
      return "Средний риск";
    case "low":
      return "Низкий риск";
    default:
      return props.riskLevel || "Риск";
  }
});

function approve(): void {
  emit("approve", props.approvalId);
}

function rejectClick(): void {
  if (!commentVisible.value) {
    commentVisible.value = true;
    return;
  }
  submitReject();
}

function submitReject(): void {
  emit("reject", { approvalId: props.approvalId, comment: comment.value.trim() });
}

/** Hotkeys must not fire while the user types in a field. */
function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

function onKeydown(e: KeyboardEvent): void {
  if (!props.hotkeys) return;
  if (isEditableTarget(e.target)) return;
  const key = e.key.toLowerCase();
  if (key === "a") {
    e.preventDefault();
    approve();
  } else if (key === "r") {
    e.preventDefault();
    rejectClick();
  }
}

onMounted(() => window.addEventListener("keydown", onKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <div class="approval" data-testid="approval">
    <p v-if="executor === 'engine'">{{ scope === "turn" ? "Разрешение действует до конца хода" : "Разрешение только для этого запроса" }}</p>
    <div class="approval-head">
      <span class="risk" :data-risk="riskLevel" data-testid="risk-label">{{ riskLabel }}</span>
      <div class="actions">
        <button
          type="button"
          class="ui-btn ui-btn--small ui-btn--outlined ui-btn--danger"
          data-testid="reject-btn"
          @click="rejectClick"
        >
          Отклонить
        </button>
        <button type="button" class="ui-btn ui-btn--small ui-btn--filled" data-testid="approve-btn" @click="approve">
          Разрешить
        </button>
      </div>
    </div>
    <div v-if="commentVisible" class="comment-field" data-testid="comment-field">
      <textarea
        v-model="comment"
        class="ui-input"
        rows="2"
        :placeholder="'Комментарий (необязательно)'"
        data-testid="reject-comment"
      />
      <div class="comment-actions">
        <button
          type="button"
          class="ui-btn ui-btn--small ui-btn--text"
          data-testid="reject-cancel"
          @click="commentVisible = false"
        >
          Отмена
        </button>
        <button type="button" class="ui-btn ui-btn--small ui-btn--filled" data-testid="reject-submit" @click="submitReject">
          Отправить
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.approval {
  margin: 0.45rem 0 0.1rem;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.approval-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 12px;
}

.risk {
  font-size: 12px;
  font-weight: 600;
}
.risk[data-risk="high"] {
  color: var(--danger);
}
.risk[data-risk="medium"] {
  color: var(--warning);
}
.risk[data-risk="low"] {
  color: var(--success);
}

.actions,
.comment-actions {
  display: flex;
  gap: 6px;
  margin-left: auto;
}

.comment-field {
  margin-top: 8px;
}

.comment-field textarea {
  resize: vertical;
  font-size: 13px;
}

.comment-actions {
  justify-content: flex-end;
  margin-top: 6px;
}
</style>

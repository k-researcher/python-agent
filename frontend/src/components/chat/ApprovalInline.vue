<script setup lang="ts">
// Inline approval controls for a pending tool call: risk level, Approve /
// Reject. Reject reveals an optional comment field. When `hotkeys` is true
// the A/R shortcuts work unless focus sits in a text input.

import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { RejectPayload } from "./types";

const props = withDefaults(
  defineProps<{ approvalId: string; riskLevel: string; hotkeys?: boolean }>(),
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
    <div class="approval-head">
      <span class="risk" :data-risk="riskLevel" data-testid="risk-label">{{ riskLabel }}</span>
      <div class="actions">
        <button type="button" class="approve" data-testid="approve-btn" @click="approve">
          Разрешить
        </button>
        <button type="button" class="reject" data-testid="reject-btn" @click="rejectClick">
          Отклонить
        </button>
      </div>
    </div>
    <div v-if="commentVisible" class="comment-field" data-testid="comment-field">
      <textarea
        v-model="comment"
        rows="2"
        :placeholder="'Комментарий (необязательно)'"
        data-testid="reject-comment"
      />
      <div class="comment-actions">
        <button type="button" class="primary" data-testid="reject-submit" @click="submitReject">
          Отправить
        </button>
        <button type="button" class="ghost" data-testid="reject-cancel" @click="commentVisible = false">
          Отмена
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.approval {
  margin: 0.45rem 0 0.1rem;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  background: var(--surface, #fff);
}

.approval-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
}

.risk {
  font-size: 0.75rem;
  font-weight: 600;
}
.risk[data-risk="high"] {
  color: var(--danger, #dc2626);
}
.risk[data-risk="medium"] {
  color: var(--warning, #d97706);
}
.risk[data-risk="low"] {
  color: var(--success, #16a34a);
}

.actions {
  display: flex;
  gap: 0.45rem;
}
.actions button {
  padding: 0.25rem 0.65rem;
  border-radius: var(--radius-sm, 6px);
  border: 1px solid var(--border, #e6e6e9);
  font-size: 0.78rem;
  cursor: pointer;
}
button.approve {
  background: var(--accent, #111114);
  color: #fff;
  border-color: var(--accent, #111114);
}
button.reject {
  background: transparent;
  color: var(--danger, #dc2626);
}

.comment-field {
  margin-top: 0.5rem;
}
.comment-field textarea {
  width: 100%;
  box-sizing: border-box;
  resize: vertical;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  padding: 0.35rem 0.5rem;
  font: inherit;
  font-size: 0.8rem;
}
.comment-actions {
  display: flex;
  gap: 0.45rem;
  margin-top: 0.4rem;
}
.comment-actions button {
  padding: 0.25rem 0.6rem;
  border-radius: var(--radius-sm, 6px);
  border: 1px solid var(--border, #e6e6e9);
  font-size: 0.78rem;
  cursor: pointer;
  background: var(--surface, #fff);
}
.comment-actions .primary {
  background: var(--accent, #111114);
  color: #fff;
  border-color: var(--accent, #111114);
}
</style>

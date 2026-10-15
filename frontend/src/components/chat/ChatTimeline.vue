<script setup lang="ts">
// Unified chat feed. Renders each TimelineItem by its `kind`, keeps the view
// pinned to the bottom while the user is already there, and shows compacted
// context checkpoints (`summary`) as muted, collapsible cards.
//
// Hotkeys (A/R) are enabled only on the first tool card that awaits approval —
// the spec routes the shortcuts to a single, unambiguous approval target.

import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { TimelineItem } from "./types";
import MarkdownView from "./MarkdownView.vue";
import MessageItem from "./MessageItem.vue";
import ReasoningBlock from "./ReasoningBlock.vue";
import ToolCallCard from "./ToolCallCard.vue";

const props = defineProps<{ items: TimelineItem[] }>();

const scroller = ref<HTMLElement | null>(null);
const summaryOpen = ref<Set<number>>(new Set());

let pinnedToBottom = true;
let resizeObserver: ResizeObserver | null = null;

/** Index (into `items`) of the first tool call awaiting approval, if any. */
const firstPendingToolIndex = computed<number | null>(() => {
  for (let i = 0; i < props.items.length; i++) {
    const item = props.items[i];
    if (item.kind === "tool" && item.call.status === "awaiting_approval") return i;
  }
  return null;
});

function onScroll(): void {
  const el = scroller.value;
  if (!el) return;
  const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
  pinnedToBottom = distance < 80;
}

function scrollToBottom(): void {
  const el = scroller.value;
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

function scrollToBottomIfPinned(): void {
  if (pinnedToBottom && scroller.value) {
    scrollToBottom();
  }
}

/**
 * Any asynchronous content growth (shiki highlighting, streaming text) changes
 * the scroller's height after the items watcher already ran. A ResizeObserver
 * re-pins the view AFTER the height settles so a highlighted code block does
 * not push the read position up.
 */
function observeScroller(): void {
  const el = scroller.value;
  if (!el || typeof ResizeObserver === "undefined") return;
  resizeObserver = new ResizeObserver(() => {
    // Re-check after the new layout is applied (next frame keeps it cheap and
    // avoids fighting intermediate heights during streaming).
    requestAnimationFrame(scrollToBottomIfPinned);
  });
  resizeObserver.observe(el);
}

watch(
  () => props.items,
  () => {
    void nextTick(scrollToBottomIfPinned);
  },
  { deep: false, immediate: true },
);

onMounted(() => observeScroller());

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  resizeObserver = null;
});

function toggleSummary(index: number): void {
  const next = new Set(summaryOpen.value);
  if (next.has(index)) next.delete(index);
  else next.add(index);
  summaryOpen.value = next;
}

function isSummaryOpen(index: number): boolean {
  return summaryOpen.value.has(index);
}
</script>

<template>
  <div ref="scroller" class="timeline" data-testid="timeline" @scroll="onScroll">
    <div class="feed">
      <template v-for="(item, index) in items" :key="itemKey(item, index)">
        <div v-if="item.kind === 'user' || item.kind === 'assistant'" class="feed-item">
          <MessageItem :message="item.message" />
        </div>

        <div v-else-if="item.kind === 'tool'" class="feed-item">
          <ToolCallCard
            :call="item.call"
            :hotkeys="firstPendingToolIndex === index"
          />
        </div>

        <div v-else-if="item.kind === 'streaming'" class="feed-item streaming" data-testid="streaming-item">
          <ReasoningBlock v-if="item.reasoning" :text="item.reasoning" :streaming="true" />
          <MarkdownView v-if="item.content" :source="item.content" />
          <span v-else-if="!item.reasoning" class="streaming-cursor" aria-hidden="true">…</span>
        </div>

        <div v-else-if="item.kind === 'summary'" class="feed-item">
          <div class="summary-card" data-testid="summary-card">
            <button
              type="button"
              class="summary-toggle"
              :aria-expanded="isSummaryOpen(index)"
              data-testid="summary-toggle"
              @click="toggleSummary(index)"
            >
              <span class="caret" :class="{ open: isSummaryOpen(index) }">▸</span>
              <span>Сводка ранней истории</span>
            </button>
            <div v-if="isSummaryOpen(index)" class="summary-body" data-testid="summary-body">
              <MarkdownView :source="item.message.content" />
            </div>
          </div>
        </div>
      </template>
      <div v-if="items.length === 0" class="empty-hint" data-testid="timeline-empty">
        Соберите ленту — сообщений пока нет.
      </div>
    </div>
  </div>
</template>

<script lang="ts">
// Helper to build a stable key per timeline entry. Lives in the plain script
// block: the import below is hoisted into the shared module scope by the
// `<script setup>` compiler pass, so `TimelineItem` is visible here too.
function itemKey(item: TimelineItem, index: number): string {
  const base = item.kind;
  if ((item.kind === "user" || item.kind === "assistant" || item.kind === "summary") && item.message?.id) {
    return `${base}:${item.message.id}`;
  }
  if (item.kind === "tool" && item.call?.id) {
    return `${base}:${item.call.id}`;
  }
  if (item.kind === "streaming") {
    return base;
  }
  return `${base}:${index}`;
}
</script>

<style scoped>
.timeline {
  flex: 1;
  overflow-y: auto;
  padding: 1rem 1.2rem 1.5rem;
}

.feed {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  max-width: 860px;
  margin: 0 auto;
}

.streaming-cursor {
  color: var(--text-muted, #6b6b76);
  animation: pulse 1.1s ease-in-out infinite;
}

.summary-card {
  margin: 0.5rem 0;
  padding: 0.45rem 0.6rem;
  border: 1px dashed var(--border, #e6e6e9);
  border-radius: var(--radius-sm, 6px);
  color: var(--text-muted, #6b6b76);
  background: var(--bg-subtle, #f4f4f6);
}

.summary-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--text-muted, #6b6b76);
  font-size: 0.78rem;
  cursor: pointer;
}

.summary-body {
  margin-top: 0.4rem;
  color: var(--text-muted, #6b6b76);
  font-size: 0.85rem;
}

.empty-hint {
  color: var(--text-muted, #6b6b76);
  text-align: center;
  padding: 2rem 0;
  font-size: 0.85rem;
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

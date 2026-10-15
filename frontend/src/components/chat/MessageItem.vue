<script setup lang="ts">
// One chat message bubble. User messages sit on the right on a subtle
// background; assistant messages on the left with reasoning collapsed below.

import { computed } from "vue";
import type { ChatMessage } from "./types";
import MarkdownView from "./MarkdownView.vue";
import ReasoningBlock from "./ReasoningBlock.vue";
import { reasoningLabel } from "./reasoning";

const props = defineProps<{ message: ChatMessage }>();

const isUser = computed(() => props.message.role === "user");

const formattedTime = computed(() => {
  const raw = props.message.created_at;
  if (!raw) return "";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
});

const effortLabel = computed(() => {
  const effort = props.message.reasoning_effort;
  if (!effort) return "";
  const label = reasoningLabel(effort);
  return effort === "none" ? label : `Рассуждения: ${label}`;
});
</script>

<template>
  <div class="message-row" :class="{ user: isUser }" data-testid="message">
    <div class="bubble" :class="{ user: isUser }">
      <MarkdownView v-if="message.content" :source="message.content" />
      <ReasoningBlock
        v-if="message.reasoning_content"
        :text="message.reasoning_content"
        :streaming="false"
      />
      <div v-if="!isUser" class="meta" data-testid="message-meta">
        <span v-if="formattedTime">{{ formattedTime }}</span>
        <span v-if="effortLabel" class="effort">{{ effortLabel }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.message-row {
  display: flex;
  width: 100%;
}
.message-row.user {
  justify-content: flex-end;
}

.bubble {
  max-width: 86%;
  padding: 0.55rem 0.7rem;
  border-radius: var(--radius, 10px);
}
.bubble.user {
  background: var(--bg-subtle, #f4f4f6);
  border-top-right-radius: 2px;
}
.bubble:not(.user) {
  background: transparent;
}

.meta {
  margin-top: 0.35rem;
  display: flex;
  gap: 0.6rem;
  color: var(--text-muted, #6b6b76);
  font-size: 0.72rem;
}
</style>

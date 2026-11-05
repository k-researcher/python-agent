<script setup lang="ts">
// One chat message bubble. User messages sit on the right on a subtle
// background; assistant messages on the left with reasoning collapsed below.

import { computed } from "vue";
import type { ChatMessage } from "./types";
import MarkdownView from "./MarkdownView.vue";
import ReasoningBlock from "./ReasoningBlock.vue";
import { reasoningLabel } from "./reasoning";
import { formatLocalTime } from "../../time";
import { FileText } from "lucide-vue-next";
import { formatSize } from "./attachments";

const props = defineProps<{ message: ChatMessage }>();

const isUser = computed(() => props.message.role === "user");

const formattedTime = computed(() => formatLocalTime(props.message.created_at));

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
      <ul v-if="message.attachments?.length" class="files" aria-label="Приложенные файлы" data-testid="message-files">
        <li v-for="(file, index) in message.attachments" :key="`${index}:${file.name}`" class="file">
          <FileText :size="13" aria-hidden="true" />
          <span class="file__name" :title="file.name">{{ file.name }}</span>
          <span class="file__size">{{ formatSize(file.size) }}</span>
        </li>
      </ul>
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

.files {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0.35rem 0 0;
  padding: 0;
  list-style: none;
}
.files:first-child {
  margin-top: 0;
}
.file {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: 100%;
  min-width: 0;
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  font-size: 0.75rem;
}
.file__name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.file__size {
  flex-shrink: 0;
  color: var(--text-muted);
}

.meta {
  margin-top: 0.35rem;
  display: flex;
  gap: 0.6rem;
  color: var(--text-muted, #6b6b76);
  font-size: 0.72rem;
}
</style>

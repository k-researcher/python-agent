<script setup lang="ts">
// The header of every settings page: back, the page title and the sections of the settings.
// One component keeps the sections in the same order and look on every page.

import { ChevronLeft } from "lucide-vue-next";
import { useRouter } from "vue-router";

defineProps<{ title: string }>();
const router = useRouter();

const SECTIONS = [
  { name: "settings-models", label: "Модели" },
  { name: "settings-speech", label: "Голос" },
  { name: "settings-skills", label: "Навыки" },
  { name: "settings-mcp", label: "MCP" },
] as const;

function back(): void {
  if (window.history.length > 1) router.back();
  else void router.push({ name: "home" });
}
</script>

<template>
  <header class="ui-page__header">
    <button type="button" class="ui-btn ui-btn--text back" data-testid="settings-back" @click="back">
      <ChevronLeft :size="18" aria-hidden="true" />
      <span>Назад</span>
    </button>
    <h1 class="ui-page__title">{{ title }}</h1>
    <nav class="ui-tabs" aria-label="Разделы настроек" data-testid="settings-nav">
      <RouterLink
        v-for="item in SECTIONS"
        :key="item.name"
        :to="{ name: item.name }"
        :data-testid="`${item.name}-link`"
      >{{ item.label }}</RouterLink>
    </nav>
  </header>
</template>

<style scoped>
.back {
  padding-left: 6px;
  gap: 2px;
}

.ui-tabs {
  margin-left: auto;
}

@media (max-width: 599px) {
  .ui-tabs {
    margin-left: 0;
  }
}
</style>

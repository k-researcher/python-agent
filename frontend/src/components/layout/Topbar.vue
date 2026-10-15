<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted } from "vue";
import { CloudOff, Loader, Moon, Search, Sun } from "lucide-vue-next";

const props = defineProps<{
  connection: "connecting" | "live" | "offline";
}>();

const emit = defineEmits<{
  search: [];
  toggleTheme: [];
}>();

const CONNECTION: Record<string, { label: string }> = {
  connecting: { label: "Подключение…" },
  live: { label: "Live" },
  offline: { label: "Нет связи" },
};

const connectionInfo = computed(() => CONNECTION[props.connection]);
function onSearchClick(): void {
  emit("search");
}

function onGlobalKeydown(event: KeyboardEvent): void {
  if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "k") {
    return;
  }
  const target = event.target as HTMLElement | null;
  const tag = target?.tagName ?? "";
  if (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target?.isContentEditable
  ) {
    return;
  }
  event.preventDefault();
  emit("search");
}

onMounted(() => window.addEventListener("keydown", onGlobalKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", onGlobalKeydown));
</script>

<template>
  <header class="topbar">
    <div class="topbar__brand">
      <span class="topbar__logo">Агент</span>
    </div>

    <button
      type="button"
      class="topbar__search"
      data-search
      @click="onSearchClick"
    >
      <Search :size="15" aria-hidden="true" />
      <span>Поиск</span>
      <kbd class="topbar__kbd">⌘K</kbd>
    </button>

    <div class="topbar__spacer" />

    <div
      class="topbar__connection"
      :data-connection="connection"
      :title="connectionInfo.label"
    >
      <Loader
        v-if="connection === 'connecting'"
        class="topbar__spin"
        :size="14"
        aria-hidden="true"
      />
      <span v-else-if="connection === 'live'" class="topbar__dot" />
      <CloudOff v-else :size="14" aria-hidden="true" />
      <span>{{ connectionInfo.label }}</span>
    </div>

    <button
      type="button"
      class="topbar__theme"
      aria-label="Переключить тему"
      title="Переключить тему"
      data-theme-toggle
      @click="emit('toggleTheme')"
    >
      <Sun :size="16" aria-hidden="true" />
      <Moon :size="16" aria-hidden="true" />
    </button>
  </header>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 52px;
  padding: 0 14px;
}

.topbar__brand {
  display: flex;
  align-items: center;
  font-weight: 750;
  font-size: 18px;
  letter-spacing: -0.02em;
  white-space: nowrap;
}

.topbar__search {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  width: 280px;
  max-width: 40vw;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  color: var(--text-muted);
  text-align: left;
}

.topbar__search:hover {
  border-color: var(--text-muted);
  color: var(--text);
}

.topbar__kbd {
  margin-left: auto;
  padding: 1px 6px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface);
  font-size: 12px;
  color: var(--text-muted);
}

.topbar__spacer {
  flex: 1;
}

.topbar__connection {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--text-muted);
  white-space: nowrap;
}

.topbar__connection[data-connection="live"] {
  color: var(--success);
}

.topbar__connection[data-connection="offline"] {
  color: var(--danger);
}

.topbar__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--success) 20%, transparent);
}

.topbar__spin {
  animation: topbar-spin 1s linear infinite;
}

@keyframes topbar-spin {
  to {
    transform: rotate(360deg);
  }
}

.topbar__theme {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-muted);
}

.topbar__theme:hover {
  color: var(--text);
}

.topbar__theme .lucide-sun {
  display: inline;
}

.topbar__theme .lucide-moon {
  display: none;
}

@media (max-width: 760px) {
  .topbar__search {
    width: auto;
    flex: 1;
  }

  .topbar__connection span:last-child {
    display: none;
  }
}
</style>

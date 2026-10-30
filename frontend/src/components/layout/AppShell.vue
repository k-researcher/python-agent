<script setup lang="ts">
// Presentational shell: Topbar on top, three columns below (320 / flexible / 320).
// Panels are shown/hidden through props; on narrow screens the toggle buttons
// appear in the top bar and emit update:* so the parent controls the state.

import {
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
} from "lucide-vue-next";

import "../../styles/tokens.css";
import "../../styles/base.css";

defineSlots<{
  topbar?: () => unknown;
  left?: () => unknown;
  main?: () => unknown;
  right?: () => unknown;
}>();

const props = withDefaults(
  defineProps<{
    leftOpen?: boolean;
    rightOpen?: boolean;
  }>(),
  {
    leftOpen: true,
    rightOpen: true,
  },
);

const emit = defineEmits<{
  "update:leftOpen": [value: boolean];
  "update:rightOpen": [value: boolean];
}>();

function toggleLeft(): void {
  emit("update:leftOpen", !props.leftOpen);
}

function toggleRight(): void {
  emit("update:rightOpen", !props.rightOpen);
}
</script>

<template>
  <div class="app-shell">
    <header class="app-shell__topbar">
      <!-- Panel toggles live in the top bar, visible only on narrow screens, so they never cover content. -->
      <button
        type="button"
        class="app-shell__toggle app-shell__toggle--left"
        :aria-expanded="leftOpen"
        :aria-label="leftOpen ? 'Скрыть левую панель' : 'Показать левую панель'"
        :title="leftOpen ? 'Скрыть левую панель' : 'Показать левую панель'"
        data-toggle="left"
        @click="toggleLeft"
      >
        <component :is="leftOpen ? PanelLeftClose : PanelLeftOpen" :size="16" />
      </button>
      <div class="app-shell__topbar-content">
        <slot name="topbar" />
      </div>
      <button
        type="button"
        class="app-shell__toggle app-shell__toggle--right"
        :aria-expanded="rightOpen"
        :aria-label="rightOpen ? 'Скрыть правую панель' : 'Показать правую панель'"
        :title="rightOpen ? 'Скрыть правую панель' : 'Показать правую панель'"
        data-toggle="right"
        @click="toggleRight"
      >
        <component :is="rightOpen ? PanelRightClose : PanelRightOpen" :size="16" />
      </button>
    </header>

    <div
      class="app-shell__body"
      :style="{
        '--left-w': leftOpen ? '320px' : '0px',
        '--right-w': rightOpen ? '320px' : '0px',
      }"
    >
      <aside v-show="leftOpen" class="app-shell__panel app-shell__left">
        <slot name="left" />
      </aside>
      <main class="app-shell__panel app-shell__main">
        <slot name="main" />
      </main>
      <aside v-show="rightOpen" class="app-shell__panel app-shell__right">
        <slot name="right" />
      </aside>
    </div>

  </div>
</template>

<style scoped>
.app-shell {
  display: grid;
  grid-template-rows: 52px minmax(0, 1fr);
  height: 100vh;
  height: 100dvh;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
}

.app-shell__topbar {
  grid-row: 1;
  display: flex;
  align-items: center;
  min-width: 0;
  z-index: 20;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}

.app-shell__topbar-content {
  flex: 1;
  min-width: 0;
  align-self: stretch;
}
.app-shell__body {
  grid-row: 2;
  display: grid;
  grid-template-columns: var(--left-w) minmax(0, 1fr) var(--right-w);
  min-height: 0;
}

.app-shell__panel {
  min-width: 0;
  min-height: 0;
}

.app-shell__left {
  grid-column: 1;
  overflow-y: auto;
  background: var(--surface);
  border-right: 1px solid var(--border);
}

.app-shell__main {
  grid-column: 2;
  overflow: hidden;
}

.app-shell__right {
  grid-column: 3;
  overflow-y: auto;
  background: var(--surface);
  border-left: 1px solid var(--border);
}

.app-shell__toggle {
  display: none; /* hidden on wide screens */
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  margin: 0 6px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-muted);
}
.app-shell__toggle:hover {
  color: var(--text);
}

.app-shell__toggle--left {
  left: 6px;
}

.app-shell__toggle--right {
  right: 6px;
}

/* Below 1100px the right column hides behind the toggle button. */
@media (max-width: 1100px) {
  .app-shell__body {
    grid-template-columns: var(--left-w) minmax(0, 1fr) 0;
  }

  .app-shell__right {
    position: fixed;
    top: 52px;
    right: 0;
    bottom: 0;
    width: min(320px, 100vw);
    z-index: 25;
    box-shadow: var(--shadow);
  }

  .app-shell__toggle--right {
    display: inline-flex;
  }
}

/* Below 760px the left column hides behind the toggle button. */
@media (max-width: 760px) {
  .app-shell__body {
    grid-template-columns: 0 minmax(0, 1fr) 0;
  }

  .app-shell__left {
    position: fixed;
    top: 52px;
    left: 0;
    bottom: 0;
    width: min(320px, 100vw);
    z-index: 25;
    box-shadow: var(--shadow);
  }

  .app-shell__toggle--left {
    display: inline-flex;
  }
}
</style>

<script setup lang="ts">
// Left sidebar: projects, session tree of the selected project, footer actions.
// Presentational only — data comes via props, actions go up through emits.
// The tree uses WAI-ARIA tree semantics and roving-tabindex keyboard navigation.

import { computed, nextTick, ref, watch } from "vue";
import {
  Archive,
  ChevronRight,
  Folder,
  LogOut,
  Plus,
  Settings,
} from "lucide-vue-next";
import type { SessionNode } from "./types";
import { statusDotClass } from "./status";

const props = withDefaults(
  defineProps<{
    projects: { id: string; name: string }[];
    selectedProjectId: string | null;
    tree: SessionNode[];
    selectedSessionId: string | null;
    isCreatingProject?: boolean;
    showCreateForm?: boolean;
  }>(),
  {
    projects: () => [],
    tree: () => [],
    selectedProjectId: null,
    selectedSessionId: null,
    isCreatingProject: false,
    showCreateForm: false,
  },
);

const emit = defineEmits<{
  selectProject: [id: string];
  selectSession: [id: string];
  createProject: [{ name: string; rootPath: string }];
  newSession: [];
  openSettings: [];
  signOut: [];
  closeCreateForm: [];
  openCreateForm: [];
}>();

// --- Create-project form ---------------------------------------------------

const internalShowCreateForm = ref(false);
const newName = ref("");
const newRootPath = ref("");

watch(() => props.showCreateForm, (val) => {
  if (val) {
    internalShowCreateForm.value = true;
  } else {
    internalShowCreateForm.value = false;
    newName.value = "";
    newRootPath.value = "";
  }
}, { immediate: true });

function closeCreateForm(): void {
  if (props.isCreatingProject) return;
  emit("closeCreateForm");
}

function submitCreate(): void {
  if (props.isCreatingProject) return;
  const name = newName.value.trim();
  const rootPath = newRootPath.value.trim();
  if (!name || !rootPath) return; // reject empty/whitespace-only input
  emit("createProject", { name, rootPath });
}

// --- Tree model ------------------------------------------------------------
// Archived nodes are pulled out of the active tree and grouped under a single
// collapsed "Архив" item, so the archive never expands on its own ("архив свёрнут").

interface Split {
  active: SessionNode[];
  archived: SessionNode[];
}

/** Splits a tree into active and archived parts, never losing a node. */
function splitTree(nodes: SessionNode[]): Split {
  const active: SessionNode[] = [];
  const archived: SessionNode[] = [];
  for (const node of nodes) {
    const sub = splitTree(node.children);
    if (node.archived) {
      archived.push({ ...node, children: sub.archived });
      active.push(...sub.active); // non-archived descendants are promoted
    } else {
      active.push({ ...node, children: sub.active });
      archived.push(...sub.archived);
    }
  }
  return { active, archived };
}

interface VisNode {
  key: string;
  depth: number;
  node: SessionNode | null; // null for the archive group row
  parentKey: string | null;
  expandable: boolean;
}

const ARCHIVE_KEY = "arch";
const ARCHIVE_STATE_KEY = "g:archive";

const expansion = ref<Record<string, boolean>>({});

function stateKey(id: string): string {
  return `s:${id}`;
}

function isNodeExpanded(node: SessionNode): boolean {
  const key = stateKey(node.id);
  return key in expansion.value ? expansion.value[key]! : !node.archived;
}

function isArchiveExpanded(): boolean {
  return expansion.value[ARCHIVE_STATE_KEY] ?? false;
}

function itemExpanded(item: VisNode): boolean {
  if (!item.expandable) return false;
  return item.node ? isNodeExpanded(item.node) : isArchiveExpanded();
}

function setItemExpanded(item: VisNode, value: boolean): void {
  if (!item.expandable) return;
  const key = item.node ? stateKey(item.node.id) : ARCHIVE_STATE_KEY;
  expansion.value = { ...expansion.value, [key]: value };
}

const visible = computed<VisNode[]>(() => {
  const { active, archived } = splitTree(props.tree);
  const result: VisNode[] = [];

  const walk = (nodes: SessionNode[], depth: number, parentKey: string | null): void => {
    for (const node of nodes) {
      // Namespace top-level keys ("n:...") so a session id can never collide
      // with the reserved archive group key ("arch").
      const key = parentKey ? `${parentKey}::${node.id}` : `n:${node.id}`;
      result.push({ key, depth, node, parentKey, expandable: node.children.length > 0 });
      if (isNodeExpanded(node)) walk(node.children, depth + 1, key);
    }
  };

  walk(active, 0, null);

  if (archived.length > 0) {
    result.push({ key: ARCHIVE_KEY, depth: 0, node: null, parentKey: null, expandable: true });
    if (isArchiveExpanded()) walk(archived, 1, ARCHIVE_KEY);
  }
  return result;
});

// --- Keyboard navigation (roving tabindex) ---------------------------------

const focusedKey = ref<string | null>(null);
const treeEl = ref<HTMLElement | null>(null);

const selectedVisibleKey = computed(() => {
  if (!props.selectedSessionId) return null;
  return visible.value.find((item) => item.node?.id === props.selectedSessionId)?.key ?? null;
});

const activeKey = computed(() => {
  if (focusedKey.value && visible.value.some((item) => item.key === focusedKey.value)) {
    return focusedKey.value;
  }
  return selectedVisibleKey.value ?? visible.value[0]?.key ?? null;
});

function focusNode(key: string): void {
  focusedKey.value = key;
  void nextTick(() => {
    const nodes = treeEl.value?.querySelectorAll<HTMLElement>("[data-node-key]");
    const index = visible.value.findIndex((item) => item.key === key);
    nodes?.[index]?.focus();
  });
}

function activate(item: VisNode): void {
  if (item.expandable) setItemExpanded(item, !itemExpanded(item));
  if (item.node) emit("selectSession", item.node.id);
  focusedKey.value = item.key;
}

function onRowClick(item: VisNode): void {
  if (item.expandable) setItemExpanded(item, !itemExpanded(item));
  if (item.node) emit("selectSession", item.node.id);
  // Move the DOM focus into the tree so the keydown handler on <ul
  // keeps receiving Arrow/Enter events after a mouse click (roving tabindex).
  focusNode(item.key);
}

function onTreeKeydown(event: KeyboardEvent): void {
  const list = visible.value;
  if (list.length === 0) return;
  const index = list.findIndex((item) => item.key === activeKey.value);
  const current = index >= 0 ? index : 0;
  const item = list[current];

  switch (event.key) {
    case "ArrowDown":
      event.preventDefault();
      focusNode(list[Math.min(current + 1, list.length - 1)].key);
      return;
    case "ArrowUp":
      event.preventDefault();
      focusNode(list[Math.max(current - 1, 0)].key);
      return;
    case "ArrowRight": {
      event.preventDefault();
      if (item && item.expandable && !itemExpanded(item)) {
        setItemExpanded(item, true);
        return;
      }
      if (item && item.expandable) {
        const firstChild = list.findIndex((child) => child.parentKey === item.key);
        if (firstChild >= 0) focusNode(list[firstChild].key);
      }
      return;
    }
    case "ArrowLeft": {
      event.preventDefault();
      if (item && item.expandable && itemExpanded(item)) {
        setItemExpanded(item, false);
        return;
      }
      if (item?.parentKey) {
        const parent = list.findIndex((p) => p.key === item.parentKey);
        if (parent >= 0) focusNode(list[parent].key);
      }
      return;
    }
    case "Home":
      event.preventDefault();
      focusNode(list[0].key);
      return;
    case "End":
      event.preventDefault();
      focusNode(list[list.length - 1].key);
      return;
    case "Enter":
    case " ":
      event.preventDefault();
      if (item) activate(item);
      return;
    default:
      return;
  }
}

function isSelected(item: VisNode): boolean {
  return item.node !== null && item.node.id === props.selectedSessionId;
}
</script>

<template>
  <div class="left-sidebar">
    <section class="left-sidebar__section left-sidebar__section--projects">
      <div class="sidebar-heading">
        <span class="sidebar-heading__title">Проекты</span>
        <button
          type="button"
          class="icon-btn"
          aria-label="Новый проект"
          :aria-expanded="internalShowCreateForm"
          data-new-project
          @click="emit('openCreateForm')"
        >
          <Plus :size="16" aria-hidden="true" />
        </button>
      </div>

      <form
        v-if="internalShowCreateForm"
        class="create-project"
        data-create-form
        :aria-busy="isCreatingProject"
        @submit.prevent="submitCreate"
      >
        <input
          v-model="newName"
          class="create-project__input"
          placeholder="Название"
          aria-label="Название проекта"
          data-create-name
          :disabled="isCreatingProject"
        />
        <input
          v-model="newRootPath"
          class="create-project__input"
          placeholder="Абсолютный путь"
          aria-label="Путь проекта"
          data-create-path
          :disabled="isCreatingProject"
        />
        <div class="create-project__actions">
          <button
            type="submit"
            class="btn btn--primary"
            data-create-submit
            :disabled="isCreatingProject"
          >
            {{ isCreatingProject ? "Создание..." : "Создать" }}
          </button>
          <button
            type="button"
            class="btn"
            data-create-cancel
            :disabled="isCreatingProject"
            @click="closeCreateForm"
          >
            Отмена
          </button>
        </div>
      </form>

      <ul class="project-list">
        <li v-for="project in projects" :key="project.id">
          <button
            type="button"
            class="project-row"
            :class="{ 'project-row--selected': project.id === selectedProjectId }"
            :aria-pressed="project.id === selectedProjectId"
            :aria-current="project.id === selectedProjectId ? 'true' : undefined"
            data-project
            @click="emit('selectProject', project.id)"
          >
            <Folder :size="15" aria-hidden="true" class="project-row__icon" />
            <span class="project-row__name">{{ project.name }}</span>
          </button>
        </li>
      </ul>
      <p v-if="projects.length === 0" class="muted left-sidebar__hint">Нет проектов</p>
    </section>

    <section class="left-sidebar__section left-sidebar__section--grow">
      <div class="sidebar-heading">
        <span class="sidebar-heading__title">Сессии</span>
        <button
          type="button"
          class="icon-btn"
          aria-label="Новая сессия"
          data-new-session
          @click="emit('newSession')"
        >
          <Plus :size="16" aria-hidden="true" />
        </button>
      </div>

      <ul ref="treeEl" class="session-tree" role="tree" aria-label="Сессии" @keydown="onTreeKeydown">
        <li
          v-for="item in visible"
          :key="item.key"
          class="tree-node"
          role="treeitem"
          :aria-level="item.depth + 1"
          :aria-expanded="item.expandable ? itemExpanded(item) : undefined"
          :aria-selected="item.node ? isSelected(item) : undefined"
          :aria-current="isSelected(item) ? 'true' : undefined"
          :tabindex="activeKey === item.key ? 0 : -1"
          :data-node-key="item.key"
        >
          <div
            class="tree-row"
            :class="{ 'tree-row--selected': isSelected(item) }"
            :style="{ paddingLeft: `${8 + item.depth * 16}px` }"
            @click="onRowClick(item)"
          >
            <span
              v-if="item.expandable"
              class="tree-row__chevron"
              :class="{ 'tree-row__chevron--open': itemExpanded(item) }"
              aria-hidden="true"
            >
              <ChevronRight :size="14" />
            </span>
            <span v-else class="tree-row__chevron tree-row__chevron--spacer" aria-hidden="true" />
            <span
              v-if="item.node"
              class="status-dot"
              :class="statusDotClass(item.node.status)"
              :title="item.node.status"
              data-status-dot
            />
            <span v-else class="status-dot status-dot--archived" aria-hidden="true">
              <Archive :size="12" />
            </span>
            <span class="tree-row__label">{{ item.node ? item.node.title : "Архив" }}</span>
          </div>
        </li>
      </ul>
      <p v-if="visible.length === 0" class="muted left-sidebar__hint">Нет сессий</p>
    </section>

    <footer class="left-sidebar__footer">
      <button type="button" class="footer-btn" data-settings @click="emit('openSettings')">
        <Settings :size="15" aria-hidden="true" />
        <span>Настройки</span>
      </button>
      <button type="button" class="footer-btn footer-btn--danger" data-signout @click="emit('signOut')">
        <LogOut :size="15" aria-hidden="true" />
        <span>Выйти</span>
      </button>
    </footer>
  </div>
</template>

<style scoped>
.left-sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.left-sidebar__section {
  padding: 10px 8px 4px;
  min-height: 0;
}

/* A long project list scrolls on its own and leaves space for the sessions and the footer. */
.left-sidebar__section--projects {
  flex: 0 1 auto;
  max-height: 40%;
  overflow-y: auto;
}

.left-sidebar__section--grow {
  flex: 1 1 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.sidebar-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 6px 6px;
}

.sidebar-heading__title {
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: none;
  color: var(--text-muted);
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  padding: 0;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
}

.icon-btn:hover {
  background: var(--bg-subtle);
  color: var(--text);
}

.create-project {
  display: grid;
  gap: 6px;
  padding: 6px;
  margin-bottom: 6px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
}

.project-list {
  display: grid;
  gap: 1px;
}

.project-row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 8px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text);
  text-align: left;
}

.project-row:hover {
  background: var(--bg-subtle);
}

.project-row--selected {
  background: var(--bg-subtle);
  font-weight: 600;
}

.project-row__icon {
  flex: none;
  color: var(--text-muted);
}

.project-row__name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.left-sidebar__hint {
  padding: 6px 8px;
  font-size: 13px;
}

.session-tree {
  display: grid;
  gap: 1px;
  padding: 0;
  margin: 0;
  list-style: none;
}

.tree-node {
  min-width: 0;
}

.tree-row {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  padding: 5px 6px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  color: var(--text);
}

.tree-row:hover,
.tree-row--selected {
  background: var(--bg-subtle);
}

.tree-row--selected {
  font-weight: 600;
}

.tree-row__chevron {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  color: var(--text-muted);
  transition: transform 0.12s ease;
}

.tree-row__chevron--open {
  transform: rotate(90deg);
}

.tree-row__chevron--spacer {
  opacity: 0;
}

.tree-row__label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status-dot {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-muted);
}

.status-dot--running {
  background: var(--info);
  animation: status-pulse 1.4s ease-in-out infinite;
}

.status-dot--awaiting {
  background: var(--warning);
}

.status-dot--completed {
  background: var(--success);
}

.status-dot--error {
  background: var(--danger);
}

.status-dot--archived {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  background: transparent;
  color: var(--text-muted);
}

@keyframes status-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}

@media (prefers-reduced-motion: reduce) {
  .status-dot--running {
    animation: none;
  }

  .tree-row__chevron {
    transition: none;
  }
}

.create-project__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}

.create-project__input {
  width: 100%;
  box-sizing: border-box;
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font-size: 14px;
}

.create-project__input:focus {
  outline: 2px solid var(--accent);
  outline-offset: -1px;
}

.btn {
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
}

.btn--primary {
  background: var(--accent);
  color: var(--accent-contrast);
  border-color: var(--accent);
}

.btn:disabled,
.create-project__input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.left-sidebar__footer {
  flex: none;
  padding: 8px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.footer-btn {
  display: inline-flex;
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
  width: 100%;
  padding: 6px 8px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text);
  font-size: 14px;
  text-align: left;
  cursor: pointer;
}

.footer-btn:hover {
  background: var(--bg-subtle);
}

.footer-btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}

.footer-btn--danger {
  color: var(--danger);
}

.footer-btn--danger:hover {
  background: color-mix(in srgb, var(--danger), transparent 90%);
}

.icon-btn svg {
  display: block;
}
</style>

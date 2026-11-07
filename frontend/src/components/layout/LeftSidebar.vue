<script setup lang="ts">
// Left sidebar: projects, session tree of the selected project, footer actions.
// Presentational only — data comes via props, actions go up through emits.
// The tree uses WAI-ARIA tree semantics and roving-tabindex keyboard navigation.

import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  Archive,
  BookOpen,
  ChevronRight,
  Folder,
  LogOut,
  MoreHorizontal,
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
    isDeletingProject?: boolean;
  }>(),
  {
    projects: () => [],
    tree: () => [],
    selectedProjectId: null,
    selectedSessionId: null,
    isCreatingProject: false,
    showCreateForm: false,
    isDeletingProject: false,
  },
);

const emit = defineEmits<{
  selectProject: [id: string];
  selectSession: [id: string];
  createProject: [{ name: string; rootPath: string }];
  newSession: [];
  openSettings: [section?: "skills"];
  signOut: [];
  closeCreateForm: [];
  openCreateForm: [];
  archiveSession: [id: string, archived: boolean];
  deleteSession: [id: string];
  deleteProject: [project: { id: string; name: string }];
}>();

// --- Row menus -------------------------------------------------------------
// One menu at a time: "s:<id>" for a session row, "p:<id>" for a project row.

const INACTIVE = new Set(["completed", "stopped", "error", "interrupted"]);
const openMenu = ref<string | null>(null);
const menuRoot = ref<HTMLElement | null>(null);
// The menu is placed against the window, so a scrolling list does not cut it off.
const menuStyle = ref<Record<string, string>>({});
const MENU_HEIGHT = 120;

function isInactive(node: SessionNode): boolean {
  return INACTIVE.has(node.status);
}

function toggleMenu(key: string, event: Event): void {
  if (openMenu.value === key) {
    openMenu.value = null;
    return;
  }
  openMenu.value = key;
  const button = event.currentTarget as HTMLElement;
  const rect = button.getBoundingClientRect();
  const right = `${Math.max(8, window.innerWidth - rect.right)}px`;
  // Near the bottom of the window the menu opens upwards.
  menuStyle.value = rect.bottom + MENU_HEIGHT > window.innerHeight
    ? { right, bottom: `${window.innerHeight - rect.top + 4}px` }
    : { right, top: `${rect.bottom + 4}px` };
  void nextTick(() => {
    menuRoot.value = button.closest("[data-menu-root]");
    menuRoot.value?.querySelector<HTMLElement>("[role=menuitem]:not(:disabled)")?.focus();
  });
}

function closeMenu(returnFocus = false): void {
  const root = menuRoot.value;
  openMenu.value = null;
  menuRoot.value = null;
  if (returnFocus) root?.querySelector<HTMLElement>("[aria-haspopup=menu]")?.focus();
}

// A click inside any row menu is handled by that menu; any other click closes it.
function onDocumentClick(event: MouseEvent): void {
  if (openMenu.value && !(event.target as Element | null)?.closest?.("[data-menu-root]")) closeMenu();
}

// A fixed menu would stay behind when its row scrolls away, so scrolling closes it.
function onViewportChange(event: Event): void {
  if (openMenu.value && !(event.target instanceof Element && event.target.closest("[role=menu]"))) closeMenu();
}

onMounted(() => {
  document.addEventListener("click", onDocumentClick);
  window.addEventListener("scroll", onViewportChange, true);
  window.addEventListener("resize", onViewportChange);
});
onBeforeUnmount(() => {
  document.removeEventListener("click", onDocumentClick);
  window.removeEventListener("scroll", onViewportChange, true);
  window.removeEventListener("resize", onViewportChange);
});

function sessionAction(node: SessionNode, action: "archive" | "delete"): void {
  closeMenu();
  if (action === "delete") emit("deleteSession", node.id);
  else emit("archiveSession", node.id, !node.archived);
}

// --- Delete-project confirmation ---------------------------------------------

const projectToDelete = ref<{ id: string; name: string } | null>(null);
const typedName = ref("");

function askDeleteProject(project: { id: string; name: string }): void {
  closeMenu();
  projectToDelete.value = { id: project.id, name: project.name };
  typedName.value = "";
}

function cancelDeleteProject(): void {
  if (props.isDeletingProject) return;
  projectToDelete.value = null;
}

function confirmDeleteProject(): void {
  const project = projectToDelete.value;
  if (!project || typedName.value !== project.name || props.isDeletingProject) return;
  emit("deleteProject", project);
}

// The form closes by itself once the project is gone from the list.
watch(() => props.projects, (projects) => {
  if (projectToDelete.value && !projects.some((project) => project.id === projectToDelete.value?.id)) {
    projectToDelete.value = null;
  }
});

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
            type="button"
            class="ui-btn ui-btn--text"
            data-create-cancel
            :disabled="isCreatingProject"
            @click="closeCreateForm"
          >
            Отмена
          </button>
          <button
            type="submit"
            class="ui-btn ui-btn--filled"
            data-create-submit
            :disabled="isCreatingProject"
          >
            {{ isCreatingProject ? "Создаю…" : "Создать" }}
          </button>
        </div>
      </form>

      <ul class="project-list">
        <li v-for="project in projects" :key="project.id" class="project-item" :data-project-item="project.id">
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
          <span class="row-menu" data-menu-root @keydown.esc.stop="closeMenu(true)">
            <button
              type="button"
              class="row-menu__button"
              :class="{ 'row-menu__button--open': openMenu === `p:${project.id}` }"
              :aria-label="`Действия с проектом «${project.name}»`"
              aria-haspopup="menu"
              :aria-expanded="openMenu === `p:${project.id}`"
              data-project-menu
              @click="toggleMenu(`p:${project.id}`, $event)"
            >
              <MoreHorizontal :size="15" aria-hidden="true" />
            </button>
            <span v-if="openMenu === `p:${project.id}`" class="row-menu__list" role="menu" :style="menuStyle">
              <button
                type="button"
                role="menuitem"
                class="row-menu__item row-menu__item--danger"
                data-project-delete
                @click="askDeleteProject(project)"
              >Удалить проект…</button>
            </span>
          </span>
        </li>
      </ul>
      <form
        v-if="projectToDelete"
        class="create-project delete-project"
        data-delete-project-form
        :aria-busy="isDeletingProject"
        @submit.prevent="confirmDeleteProject"
      >
        <p class="delete-project__text">
          Удалить проект «{{ projectToDelete.name }}»? Удалятся все его сессии и история.
          Файлы в папке проекта останутся.
        </p>
        <input
          v-model="typedName"
          class="create-project__input"
          :placeholder="projectToDelete.name"
          aria-label="Введите имя проекта для подтверждения"
          autocomplete="off"
          data-delete-project-name
          :disabled="isDeletingProject"
        />
        <div class="create-project__actions">
          <button
            type="button"
            class="ui-btn ui-btn--text"
            :disabled="isDeletingProject"
            @click="cancelDeleteProject"
          >Отмена</button>
          <button
            type="submit"
            class="ui-btn ui-btn--filled ui-btn--danger"
            data-delete-project-submit
            :disabled="typedName !== projectToDelete.name || isDeletingProject"
          >
            {{ isDeletingProject ? "Удаляю…" : "Удалить проект" }}
          </button>
        </div>
      </form>
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
            <span
              v-if="item.node"
              class="row-menu"
              data-menu-root
              @click.stop
              @keydown.stop
              @keydown.esc="closeMenu(true)"
            >
              <button
                type="button"
                class="row-menu__button"
                :class="{ 'row-menu__button--open': openMenu === `s:${item.node.id}` }"
                :tabindex="activeKey === item.key ? 0 : -1"
                :aria-label="`Действия с сессией «${item.node.title}»`"
                aria-haspopup="menu"
                :aria-expanded="openMenu === `s:${item.node.id}`"
                data-session-menu
                @click="toggleMenu(`s:${item.node.id}`, $event)"
              >
                <MoreHorizontal :size="15" aria-hidden="true" />
              </button>
              <span v-if="openMenu === `s:${item.node.id}`" class="row-menu__list" role="menu" :style="menuStyle">
                <button
                  type="button"
                  role="menuitem"
                  class="row-menu__item"
                  :disabled="!item.node.archived && !isInactive(item.node)"
                  data-session-archive
                  @click="sessionAction(item.node, 'archive')"
                >{{ item.node.archived ? "Вернуть из архива" : "В архив" }}</button>
                <button
                  type="button"
                  role="menuitem"
                  class="row-menu__item row-menu__item--danger"
                  :disabled="!isInactive(item.node)"
                  data-session-delete
                  @click="sessionAction(item.node, 'delete')"
                >Удалить</button>
                <span v-if="!isInactive(item.node)" class="row-menu__hint">Сначала остановите сессию.</span>
              </span>
            </span>
          </div>
        </li>
      </ul>
      <p v-if="visible.length === 0" class="muted left-sidebar__hint">Нет сессий</p>
    </section>

    <footer class="left-sidebar__footer">
      <button type="button" class="footer-btn" data-skills @click="emit('openSettings', 'skills')">
        <BookOpen :size="15" aria-hidden="true" />
        <span>Навыки</span>
      </button>
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
  overflow-x: hidden;
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
  padding: 8px;
  margin-bottom: 6px;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
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
  justify-content: flex-end;
  gap: 6px;
  margin-top: 4px;
}

.create-project__input {
  width: 100%;
  box-sizing: border-box;
  min-height: var(--control-height);
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--text);
  font-size: 14px;
}

.create-project__input:focus {
  outline: 2px solid var(--accent);
  outline-offset: -1px;
}

.delete-project__text {
  margin: 0;
  font-size: 13px;
  color: var(--text);
}

/* Row menu: shown on hover, focus, for the selected row and always on touch screens. */
.project-item {
  display: flex;
  align-items: center;
  gap: 2px;
}

.project-item .project-row {
  flex: 1 1 auto;
  min-width: 0;
}

.row-menu {
  position: relative;
  flex: none;
  margin-left: auto;
}

.row-menu__button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
  opacity: 0;
  cursor: pointer;
}

.row-menu__button:hover {
  background: var(--surface);
  color: var(--text);
}

.tree-row:hover .row-menu__button,
.tree-row--selected .row-menu__button,
.project-item:hover .row-menu__button,
.row-menu__button:focus-visible,
.row-menu__button--open {
  opacity: 1;
}

@media (hover: none) {
  .row-menu__button {
    opacity: 1;
  }
}

/* The place comes from the script (menuStyle); the corners follow the menu: outer 10, inner 6. */
.row-menu__list {
  position: fixed;
  z-index: 50;
  display: grid;
  min-width: 180px;
  padding: 4px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow);
  font-weight: 400;
}

.row-menu__item {
  min-height: 32px;
  padding: 6px 10px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text);
  font-size: 13px;
  text-align: left;
  white-space: nowrap;
  cursor: pointer;
}

.row-menu__item:hover:not(:disabled),
.row-menu__item:focus-visible {
  background: var(--bg-subtle);
}

.row-menu__item:disabled {
  opacity: 0.5;
  cursor: default;
}

.row-menu__item--danger {
  color: var(--danger);
}

.row-menu__hint {
  padding: 2px 8px 4px;
  color: var(--text-muted);
  font-size: 12px;
  white-space: nowrap;
}

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

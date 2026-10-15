<script setup lang="ts">
// New app shell: auth, realtime, theme, sidebar persistence, layout wiring.

import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useAppStore } from "./stores/app";
import { useSessionStore } from "./stores/session";
import { Realtime, type ConnectionStatus } from "./realtime";
import { clearAuth, loginFromFragment, logout, refreshAuth } from "./auth";
import { confirmAuthenticated, newSessionProfile, resetAuthCache } from "./router";
import type { ServerEvent } from "./api/types";
import AppShell from "./components/layout/AppShell.vue";
import Topbar from "./components/layout/Topbar.vue";
import LeftSidebar from "./components/layout/LeftSidebar.vue";
import RightPanel from "./components/layout/RightPanel.vue";
import {
  toChildSessions,
  toAuditEntries,
  toLayoutContext,
  toLayoutSessionTree,
  toProjectOptions,
  toToolEntries,
} from "./uiAdapters";

const appStore = useAppStore();
const sessionStore = useSessionStore();
const router = useRouter();
const route = useRoute();

// ---------------------------------------------------------------------------
// Auth state
// ---------------------------------------------------------------------------

const authenticated = ref<boolean | null>(null);
let authGeneration = 0;

// Surface-level error/notice shown in the shell (create-project failures,
// settings explanation) so a failed action is never swallowed silently.
const projectError = ref("");
const shellNotice = ref("");

// ---------------------------------------------------------------------------
// Theme (light by default; preference in localStorage with guarded reads)
// ---------------------------------------------------------------------------

const THEME_KEY = "agent:theme";

function readStoredTheme(): "light" | "dark" {
  try {
    const saved = window.localStorage.getItem(THEME_KEY);
    return saved === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

function writeStoredTheme(theme: "light" | "dark"): void {
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // storage unavailable — theme just won't persist
  }
}

const theme = ref<"light" | "dark">(readStoredTheme());

function applyTheme(t: "light" | "dark"): void {
  document.documentElement.setAttribute("data-theme", t);
}

// Unconditionally set theme on startup/mount
onMounted(() => {
  applyTheme(theme.value);
});

function toggleTheme(): void {
  theme.value = theme.value === "dark" ? "light" : "dark";
  applyTheme(theme.value);
  writeStoredTheme(theme.value);
}

// ---------------------------------------------------------------------------
// Sidebar state (persisted separately from theme)
// ---------------------------------------------------------------------------

const SIDEBAR_KEY = "agent:sidebar";
const leftOpen = ref(window.innerWidth > 760);
const rightOpen = ref(window.innerWidth > 1100);

function readSidebarState(): void {
  try {
    const saved = window.localStorage.getItem(SIDEBAR_KEY);
    if (!saved) return;
    const parsed: unknown = JSON.parse(saved);
    if (typeof parsed !== "object" || parsed === null) return;
    const rec = parsed as Record<string, unknown>;
    if (typeof rec.leftOpen === "boolean") leftOpen.value = rec.leftOpen;
    if (typeof rec.rightOpen === "boolean") rightOpen.value = rec.rightOpen;
  } catch {
    // ignore corrupt saved state
  }
}

watch(
  [leftOpen, rightOpen],
  () => {
    try {
      window.localStorage.setItem(
        SIDEBAR_KEY,
        JSON.stringify({ leftOpen: leftOpen.value, rightOpen: rightOpen.value }),
      );
    } catch {
      // storage unavailable — sidebar state won't persist
    }
  },
);

// ---------------------------------------------------------------------------
// Realtime
// ---------------------------------------------------------------------------

let realtime: Realtime | null = null;
let wasConnected = false;
const connection = ref<ConnectionStatus>("offline");

function onServerEvent(event: ServerEvent): void {
  if (authenticated.value !== true) return;
  appStore.applyEvent(event);
  sessionStore.applyEvent(event);
}

function onRealtimeStatus(status: ConnectionStatus): void {
  if (authenticated.value !== true) return;
  connection.value = status;
  if (status === "live") {
    // Initial connect or reconnect: reload the stores so we catch up to
    // events that may have fired while the socket was down. Reopening the
    // current session re-fetches messages/approvals/context consistently.
    const reload = async (): Promise<void> => {
      const expectedGeneration = authGeneration;
      try {
        await appStore.load();
        if (expectedGeneration !== authGeneration || authenticated.value !== true) return;
        if (sessionStore.sessionId) {
          const id = sessionStore.sessionId;
          await sessionStore.open(id);
        }
      } catch {
        // transient failure — data stays as-is, next reconnect will retry
      }
    };
    void reload();
    wasConnected = true;
  }
}

function connectRealtime(): void {
  if (realtime) {
    realtime.close();
    realtime = null;
  }
  wasConnected = false;
  const expectedGeneration = authGeneration;
  realtime = new Realtime({
    onEvent: (event) => {
      if (expectedGeneration === authGeneration) onServerEvent(event);
    },
    onStatus: (status) => {
      if (expectedGeneration === authGeneration) onRealtimeStatus(status);
    },
    onUnauthorized: () => {
      if (expectedGeneration === authGeneration) endAuthSession();
    },
  });
  realtime.connect();
}

// ---------------------------------------------------------------------------
// Layout data from stores → component props via adapters
// ---------------------------------------------------------------------------

const projectOptions = computed(() => toProjectOptions(appStore.projects));
const sessionTree = computed(() =>
  toLayoutSessionTree(appStore.sessionTree(appStore.selectedProjectId)),
);
const selectedSessionId = computed(() => sessionStore.sessionId);
const contextStats = computed(() => toLayoutContext(sessionStore.context));
const childSessions = computed(() => toChildSessions(sessionStore.children));
const auditEntries = computed(() => toAuditEntries(sessionStore.audit));
const toolEntries = computed(() => toToolEntries(appStore.config));

// ---------------------------------------------------------------------------
// Sidebar actions
// ---------------------------------------------------------------------------

async function onSelectProject(id: string): Promise<void> {
  if (id === appStore.selectedProjectId || !appStore.projects.some((project) => project.id === id)) return;
  sessionStore.close();
  appStore.selectedProjectId = id;
  projectError.value = "";
  shellNotice.value = "";
  void router.push({ name: "home" });
}

watch(() => sessionStore.session?.project_id, (projectId) => {
  if (projectId && appStore.projects.some((project) => project.id === projectId)) {
    appStore.selectedProjectId = projectId;
  }
});

function onSelectSession(id: string): void {
  if (window.innerWidth <= 760) leftOpen.value = false;
  void router.push(`/s/${id}`);
}

// Track if a project creation is in progress to disable UI
const isCreatingProject = ref(false);
const isCreateFormVisible = ref(false);

async function onCreateProject(input: { name: string; rootPath: string }): Promise<void> {
  if (isCreatingProject.value) return;
  projectError.value = "";
  isCreatingProject.value = true;
  const expectedGeneration = authGeneration;
  try {
    await appStore.createProject(input.name, input.rootPath);
    if (expectedGeneration !== authGeneration) return;
    // On success, close the form.
    isCreateFormVisible.value = false;
  } catch (cause) {
    if (expectedGeneration !== authGeneration) return;
    // Surface the failure instead of swallowing it: the user must know that the
    // project was not created (the form stays open with its values untouched).
    projectError.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (expectedGeneration === authGeneration) isCreatingProject.value = false;
  }
}

function onNewSession(): void {
  sessionStore.close();
  if (window.innerWidth <= 760) leftOpen.value = false;
  void router.push({ name: "home" });
}

function onOpenSettings(): void {
  // Settings / model-management (3.3) is not implemented yet. Be honest about it
  // rather than navigating somewhere that silently does nothing.
  shellNotice.value =
    "Настройки и управление моделями будут доступны в следующем этапе (3.3).";
}

function endAuthSession(): void {
  authGeneration += 1;
  authenticated.value = false;
  resetAuthCache();
  clearAuth();
  newSessionProfile.value = null;
  realtime?.close();
  realtime = null;
  connection.value = "offline";
  wasConnected = false;
  sessionStore.close();
  appStore.reset();
  projectError.value = "";
  shellNotice.value = "";
  isCreatingProject.value = false;
  isCreateFormVisible.value = false;
  void router.replace({ name: "login" });
}

async function onSignOut(): Promise<void> {
  const request = logout();
  endAuthSession();
  try {
    await request;
  } catch {
    return;
  }
}

// ---------------------------------------------------------------------------
// RightPanel navigation
// ---------------------------------------------------------------------------

function onOpenChildSession(id: string): void {
  void router.push(`/s/${id}`);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function bootstrap(): Promise<void> {
  const expectedGeneration = authGeneration;
  try {
    await appStore.load();
  } catch {
    // If loading fails (network), the UI still renders; sidebars will retry on
    // the next realtime reconnect.
  }
  if (expectedGeneration === authGeneration && authenticated.value === true) connectRealtime();
}

async function enterAuthenticatedWorkspace(): Promise<void> {
  const expectedGeneration = authGeneration;
  authenticated.value = true;
  confirmAuthenticated();
  if (route.name === "login") {
    const next = route.query.next;
    const destination = typeof next === "string" && next.startsWith("/")
      && !next.startsWith("//") && next.split(/[?#]/)[0] !== "/login"
      ? next : { name: "home" };
    await router.replace(destination);
  }
  if (expectedGeneration === authGeneration) await bootstrap();
}

async function init(): Promise<void> {
  const expectedGeneration = authGeneration;
  readSidebarState();

  // Check for a one-time login link first.
  let state = null;
  try {
    state = await loginFromFragment();
  } catch {
    // Invalid or expired link; fall through to regular check.
  }
  if (expectedGeneration !== authGeneration) return;
  if (state && state.authenticated) {
    await enterAuthenticatedWorkspace();
    return;
  }
  if (state) {
    // Link was consumed but not authenticated.
    authenticated.value = false;
    resetAuthCache();
    void router.replace({ name: "login" });
    return;
  }

  // Regular session check.
  try {
    const session = await refreshAuth();
    if (expectedGeneration !== authGeneration) return;
    authenticated.value = session.authenticated;
  } catch {
    if (expectedGeneration !== authGeneration) return;
    authenticated.value = false;
  }
  if (authenticated.value) {
    await enterAuthenticatedWorkspace();
  } else {
    resetAuthCache();
    if (route.name !== "login") void router.replace({ name: "login" });
  }
}

onMounted(() => {
  void init();
});

onBeforeUnmount(() => {
  authGeneration += 1;
  appStore.reset();
  sessionStore.close();
  realtime?.close();
  realtime = null;
});
</script>

<template>
  <div v-if="authenticated === null" class="app-loading">
    <p>Загрузка…</p>
  </div>

  <div v-else-if="authenticated === false" class="app-login">
    <RouterView />
  </div>

  <AppShell
    v-else
    v-model:left-open="leftOpen"
    v-model:right-open="rightOpen"
  >
    <template #topbar>
      <Topbar
        :connection="connection"
        @toggle-theme="toggleTheme"
        @search="() => {}"
      />
    </template>

    <template #left>
      <LeftSidebar
        :projects="projectOptions"
        :selected-project-id="appStore.selectedProjectId"
        :tree="sessionTree"
        :selected-session-id="selectedSessionId"
        :is-creating-project="isCreatingProject"
        :show-create-form="isCreateFormVisible"
        @select-project="onSelectProject"
        @select-session="onSelectSession"
        @create-project="onCreateProject"
        @new-session="onNewSession"
        @open-settings="onOpenSettings"
        @sign-out="onSignOut"
        @open-create-form="isCreateFormVisible = true"
        @close-create-form="isCreateFormVisible = false"
      />
    </template>

    <template #main>
      <div class="shell-main">
        <div v-if="projectError || shellNotice" class="shell-banner" role="status">
          <p v-if="projectError" class="shell-banner__error" data-testid="project-error">
            {{ projectError }}
          </p>
          <p v-else-if="shellNotice" class="shell-banner__notice" data-testid="shell-notice">
            {{ shellNotice }}
          </p>
        </div>
        <div class="shell-main__view">
          <RouterView />
        </div>
      </div>
    </template>

    <template #right>
      <RightPanel
        :context="contextStats"
        :children="childSessions"
        :audit="auditEntries"
        :tools="toolEntries"
        @open-session="onOpenChildSession"
      />
    </template>
  </AppShell>
</template>

<style scoped>
.app-loading,
.app-login {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  color: var(--text-muted);
  font-size: 14px;
}

.shell-main {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.shell-main__view {
  flex: 1;
  min-height: 0;
}

.shell-banner {
  flex-shrink: 0;
  margin: 8px 16px 0;
}

.shell-banner__error {
  margin: 0;
  padding: 8px 12px;
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  color: var(--danger);
  font-size: 13px;
}

.shell-banner__notice {
  margin: 0;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  color: var(--text-muted);
  font-size: 13px;
}
</style>

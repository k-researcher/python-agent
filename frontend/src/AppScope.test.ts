import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { nextTick } from "vue";
import App from "./App.vue";
import { api } from "./api/client";
import { clearAuth, loginFromFragment, logout, refreshAuth } from "./auth";
import { confirmAuthenticated, resetAuthCache } from "./router";
import { Realtime, type RealtimeOptions } from "./realtime";
import { useAppStore } from "./stores/app";
import { useSessionStore } from "./stores/session";
import type { Project } from "./api/types";

const navigation = vi.hoisted(() => ({
  push: vi.fn(), replace: vi.fn(),
  route: { name: "home", query: {} as Record<string, string> },
}));

vi.mock("./auth", () => ({
  clearAuth: vi.fn(), loginFromFragment: vi.fn().mockResolvedValue(null),
  refreshAuth: vi.fn().mockResolvedValue({ authenticated: true }),
  logout: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("./router", () => ({
  confirmAuthenticated: vi.fn(), resetAuthCache: vi.fn(), newSessionProfile: { value: null },
}));
vi.mock("vue-router", () => ({
  useRouter: () => navigation, useRoute: () => navigation.route,
}));
vi.mock("./realtime", () => ({
  Realtime: vi.fn(function MockRealtime() { return { connect: vi.fn(), close: vi.fn() }; }),
}));

const projects: Project[] = [
  { id: "first", name: "First", root_path: "/first" },
  { id: "second", name: "Second", root_path: "/second" },
];

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>((complete) => { resolve = complete; });
  return { promise, resolve };
}

const wrappers: ReturnType<typeof mount>[] = [];

async function readyApp() {
  const wrapper = mount(App, {
    global: {
      stubs: {
        AppShell: { template: '<div><slot name="left" /><slot name="main" /><slot name="right" /></div>' },
        LeftSidebar: {
          emits: ["sign-out", "select-project", "new-session"],
          template: `<div>
            <button data-sign-out @click="$emit('sign-out')">Exit</button>
            <button data-second-project @click="$emit('select-project', 'second')">Second</button>
            <button data-new-session @click="$emit('new-session')">New</button>
          </div>`,
        },
        RightPanel: {
          name: "RightPanelTestStub",
          props: ["context", "children", "audit", "tools"],
          template: "<div />",
        },
        RouterView: true,
      },
    },
  });
  wrappers.push(wrapper);
  await flushPromises();
  return wrapper;
}

function realtimeOptions(): RealtimeOptions {
  const options = vi.mocked(Realtime).mock.calls.at(-1)?.[0];
  if (!options) throw new Error("Realtime was not connected");
  return options;
}

beforeEach(() => {
  vi.clearAllMocks();
  navigation.route.name = "home";
  navigation.route.query = {};
  vi.mocked(loginFromFragment).mockResolvedValue(null);
  vi.mocked(refreshAuth).mockResolvedValue({ authenticated: true });
  vi.mocked(logout).mockResolvedValue(undefined);
  setActivePinia(createPinia());
  vi.spyOn(api, "config").mockResolvedValue({
    llm_configured: true, llm_model: "model", llm_destination: "http://local",
    knowledge_base_enabled: false, network_tools_enabled: false,
    network_allowlist: [], api_auth_enabled: true, execution_mode: "embedded",
    default_llm_profile: "default", llm_profiles: [],
  });
  vi.spyOn(api, "projects").mockResolvedValue(projects);
  vi.spyOn(api, "sessions").mockResolvedValue([]);
});

afterEach(() => {
  for (const wrapper of wrappers.splice(0)) wrapper.unmount();
  vi.restoreAllMocks();
});

describe("authenticated workspace lifecycle", () => {
  it("keeps child sessions in panel props instead of forwarding them to DOM attrs", async () => {
    const wrapper = await readyApp();
    const panel = wrapper.findComponent({ name: "RightPanelTestStub" });
    expect(panel.props("children")).toEqual([]);
    expect(panel.vm.$attrs).not.toHaveProperty("children");
  });

  it("a confirmed session primes the router instead of clearing its cache", async () => {
    await readyApp();
    expect(confirmAuthenticated).toHaveBeenCalledOnce();
    expect(resetAuthCache).not.toHaveBeenCalled();
  });

  it("a confirmed login link restores the protected destination", async () => {
    navigation.route.name = "login";
    navigation.route.query = { next: "/s/protected" };
    vi.mocked(loginFromFragment).mockResolvedValueOnce({ authenticated: true });
    await readyApp();
    expect(confirmAuthenticated).toHaveBeenCalledOnce();
    expect(resetAuthCache).not.toHaveBeenCalled();
    expect(refreshAuth).not.toHaveBeenCalled();
    expect(navigation.replace).toHaveBeenCalledWith("/s/protected");
  });

  it.each(["//external.example", "/login", "/login?next=/s/protected", "https://external.example"])(
    "a confirmed session does not restore an unsafe or login destination: %s",
    async (next) => {
      navigation.route.name = "login";
      navigation.route.query = { next };
      await readyApp();
      expect(navigation.replace).toHaveBeenCalledWith({ name: "home" });
    },
  );

  it("a confirmed refusal clears the router cache without opening the workspace", async () => {
    vi.mocked(refreshAuth).mockResolvedValueOnce({ authenticated: false });
    await readyApp();
    expect(confirmAuthenticated).not.toHaveBeenCalled();
    expect(resetAuthCache).toHaveBeenCalledOnce();
    expect(api.projects).not.toHaveBeenCalled();
    expect(navigation.replace).toHaveBeenCalledWith({ name: "login" });
  });

  it("confirmed authorization loss clears workspace data and redirects", async () => {
    await readyApp();
    const sessionStore = useSessionStore();
    sessionStore.sessionId = "private";
    realtimeOptions().onUnauthorized?.();
    expect(useAppStore().projects).toEqual([]);
    expect(useAppStore().config).toBeNull();
    expect(sessionStore.sessionId).toBeNull();
    expect(clearAuth).toHaveBeenCalledOnce();
    expect(navigation.replace).toHaveBeenCalledWith({ name: "login" });
  });

  it("sign-out clears local data before a slow server logout finishes", async () => {
    const pendingLogout = deferred<void>();
    vi.mocked(logout).mockReturnValueOnce(pendingLogout.promise);
    const wrapper = await readyApp();
    await wrapper.get("[data-sign-out]").trigger("click");
    expect(useAppStore().projects).toEqual([]);
    expect(navigation.replace).toHaveBeenCalledWith({ name: "login" });
    pendingLogout.resolve();
    await flushPromises();
  });

  it("events from the closed socket cannot trigger new workspace reloads", async () => {
    const wrapper = await readyApp();
    const callbacks = realtimeOptions();
    await wrapper.get("[data-sign-out]").trigger("click");
    const previousCalls = vi.mocked(api.projects).mock.calls.length;
    callbacks.onEvent?.({ id: 1, created_at: "2026-01-01T00:00:00Z", type: "project.created", session_id: null, payload: {} });
    callbacks.onStatus?.("live");
    await flushPromises();
    expect(api.projects).toHaveBeenCalledTimes(previousCalls);
    expect(useAppStore().projects).toEqual([]);
  });

  it("an in-flight reconnect cannot restore data or reopen a session after sign-out", async () => {
    const wrapper = await readyApp();
    const previousProjects = deferred<Project[]>();
    vi.mocked(api.projects).mockReturnValueOnce(previousProjects.promise);
    const sessionStore = useSessionStore();
    sessionStore.sessionId = "private";
    const open = vi.spyOn(sessionStore, "open");
    realtimeOptions().onStatus?.("live");
    await wrapper.get("[data-sign-out]").trigger("click");
    previousProjects.resolve(projects);
    await flushPromises();
    expect(useAppStore().projects).toEqual([]);
    expect(open).not.toHaveBeenCalled();
  });

  it("switching project closes the previous session and returns to the home view", async () => {
    const wrapper = await readyApp();
    useSessionStore().sessionId = "private";
    await wrapper.get("[data-second-project]").trigger("click");
    expect(useAppStore().selectedProjectId).toBe("second");
    expect(useSessionStore().sessionId).toBeNull();
    expect(navigation.push).toHaveBeenCalledWith({ name: "home" });
  });

  it("an opened session selects its own visible project", async () => {
    await readyApp();
    useSessionStore().session = {
      id: "session", project_id: "second", parent_id: null, title: "Task",
      mode: "dev", llm_profile: "default", status: "completed", error: null,
      archived: false, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
    };
    await nextTick();
    expect(useAppStore().selectedProjectId).toBe("second");
  });
});

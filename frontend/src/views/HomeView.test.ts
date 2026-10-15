import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createRouter, createMemoryHistory } from "vue-router";
import HomeView from "./HomeView.vue";
import { useAppStore } from "../stores/app";
import { api } from "../api/client";
import { newSessionProfile } from "../router";

function createStorage(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key: string) => (store.has(key) ? (store.get(key) as string) : null),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    removeItem: (key: string) => void store.delete(key),
    setItem: (key: string, value: string) => void store.set(key, String(value)),
  };
}

let storage: Storage | null = null;

const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: "/", component: HomeView }, { path: "/s/:id", component: { template: "<div/>" } }],
});

function prepareStore(): ReturnType<typeof useAppStore> {
  const store = useAppStore();
  store.config = {
    llm_configured: true,
    llm_model: "m",
    llm_destination: "dst",
    knowledge_base_enabled: false,
    network_tools_enabled: true,
    network_allowlist: [],
    api_auth_enabled: true,
    execution_mode: "embedded",
    default_llm_profile: "p1",
    llm_profiles: [
      { name: "p1", model: "model-a", destination: "dst", configured: true },
      { name: "p2", model: "model-b", destination: "dst", configured: false },
    ],
  };
  store.projects = [{ id: "p1", name: "Проект", root_path: "/tmp/p1" }];
  store.selectedProjectId = "p1";
  return store;
}

beforeEach(() => {
  storage = createStorage();
  Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
  setActivePinia(createPinia());
});

afterEach(() => {
  Object.defineProperty(window, "localStorage", { value: undefined, configurable: true });
  storage = null;
  newSessionProfile.value = null;
  vi.restoreAllMocks();
});

describe("HomeView", () => {
  it("groups model, role and reasoning controls inside the composer", () => {
    prepareStore();
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="model-select"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="mode-select"]').exists()).toBe(true);
    const controls = wrapper.get('[data-testid="agent-controls"]');
    expect(controls.find('[data-testid="model-select"]').exists()).toBe(true);
    expect(controls.find('[data-testid="mode-select"]').exists()).toBe(true);
    expect(controls.find('[data-testid="composer-effort"]').exists()).toBe(true);
  });

  it("emits a template picked from a suggestion card into the composer draft", async () => {
    prepareStore();
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await wrapper.findAll('[data-testid="suggestion-card"]')[0].trigger("click");
    const input = wrapper.get('[data-testid="composer-input"]');
    expect((input.element as HTMLTextAreaElement).value).toBe(
      "Предложи три интересные идеи для этого проекта.",
    );
  });

  it("creates a session on send and navigates to it", async () => {
    const store = prepareStore();
    const createSessionSpy = vi
      .spyOn(store, "createSession")
      .mockResolvedValue({
        id: "s-new",
        project_id: "p1",
        parent_id: null,
        mode: "dev",
        llm_profile: "p1",
        title: "t",
        status: "pending",
        error: null,
        archived: false,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      });
    const pushSpy = vi.spyOn(router, "push").mockResolvedValue();

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await wrapper.get('[data-testid="composer-input"]').setValue("создай отчёт");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    await flushPromises();

    expect(createSessionSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        project_id: "p1",
        prompt: "создай отчёт",
        mode: "dev",
        auto_start: true,
      }),
    );
    expect(pushSpy).toHaveBeenCalledWith("/s/s-new");
  });

  it("shows an error when no project is selected", async () => {
    const store = prepareStore();
    store.selectedProjectId = null;
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await wrapper.get('[data-testid="composer-input"]').setValue("что-то");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    expect(wrapper.get('[data-testid="home-error"]').text()).toContain("Выберите проект");
  });

  it("does not send when text is empty", async () => {
    prepareStore();
    const createSpy = vi.spyOn(useAppStore(), "createSession").mockResolvedValue({} as never);
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    expect(createSpy).not.toHaveBeenCalled();
  });

  it("restores the draft when creating a session fails", async () => {
    const store = prepareStore();
    vi.spyOn(store, "createSession").mockRejectedValue(new Error("boom"));
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await wrapper.get('[data-testid="composer-input"]').setValue("важный черновик");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    await flushPromises();

    expect(wrapper.get('[data-testid="home-error"]').text()).toContain("boom");
    // The composer still shows the restored text instead of losing it.
    expect(
      (wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value,
    ).toBe("важный черновик");
  });

  it("remembers the new-session profile in the only model picker", async () => {
    const store = prepareStore();
    // Make p2 a configured, selectable profile for this test.
    if (store.config) {
      store.config = {
        ...store.config,
        llm_profiles: store.config.llm_profiles.map((p) =>
          p.name === "p2" ? { ...p, configured: true } : p,
        ),
      };
    }
    newSessionProfile.value = "p2";
    const createSessionSpy = vi
      .spyOn(store, "createSession")
      .mockResolvedValue({
        id: "s-new",
        project_id: "p1",
        parent_id: null,
        mode: "dev",
        llm_profile: "p2",
        title: "t",
        status: "pending",
        error: null,
        archived: false,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      });
    const pushSpy = vi.spyOn(router, "push").mockResolvedValue();

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    expect((wrapper.get('[data-testid="model-select"]').element as HTMLSelectElement).value).toBe(
      "p2",
    );

    await wrapper.get('[data-testid="composer-input"]').setValue("создай отчёт");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    await flushPromises();

    expect(createSessionSpy).toHaveBeenCalledWith(
      expect.objectContaining({ llm_profile: "p2" }),
    );
    expect(pushSpy).toHaveBeenCalledWith("/s/s-new");
  });

  it("loads reasoning levels from the selected profile and resets an unsupported choice", async () => {
    const store = prepareStore();
    store.config!.llm_profiles[0]!.reasoning_efforts = ["low", "high"];
    store.config!.llm_profiles[1]!.configured = true;
    store.config!.llm_profiles[1]!.reasoning_efforts = ["medium"];
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    const effort = wrapper.get('[data-testid="composer-effort"]');
    expect(effort.findAll("option").map((option) => option.attributes("value"))).toEqual(["auto", "low", "high"]);
    await effort.setValue("high");
    await wrapper.get('[data-testid="model-select"]').setValue("p2");
    expect(effort.findAll("option").map((option) => option.attributes("value"))).toEqual(["auto", "medium"]);
    expect((effort.element as HTMLSelectElement).value).toBe("auto");
    expect(newSessionProfile.value).toBe("p2");
  });

  it("locks all creation settings and does not offer a nonfunctional stop button", async () => {
    const store = prepareStore();
    let release: (reason: Error) => void = () => {};
    vi.spyOn(store, "createSession").mockImplementation(() => new Promise((_, reject) => { release = reject; }));
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await wrapper.get('[data-testid="composer-input"]').setValue("Новая задача");
    await wrapper.get('[data-testid="composer-send"]').trigger("click");
    for (const select of wrapper.get('[data-testid="agent-controls"]').findAll("select")) {
      expect(select.attributes("disabled")).toBeDefined();
    }
    expect(wrapper.find('[data-testid="composer-stop"]').exists()).toBe(false);
    expect(wrapper.get('[data-testid="composer-send"]').attributes("disabled")).toBeDefined();
    release(new Error("Отказ"));
    await flushPromises();
    expect(wrapper.get('[data-testid="composer-input"]').attributes("disabled")).toBeUndefined();
  });
});

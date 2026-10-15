import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises, enableAutoUnmount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createRouter, createMemoryHistory } from "vue-router";
import HomeView from "./HomeView.vue";
import { useAppStore } from "../stores/app";
import { newSessionProfile } from "../router";
import type { Session } from "../api/types";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

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
const originalStorage = Object.getOwnPropertyDescriptor(window, "localStorage");

enableAutoUnmount(afterEach);

const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/", component: HomeView },
    { path: "/s/:id", component: { template: "<div></div>" } },
  ],
});

const sessionFixture: Session = {
  id: "s-123",
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
};

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
  store.projects = [
    { id: "p1", name: "Project A", root_path: "/tmp/p1" },
    { id: "p2", name: "Project B", root_path: "/tmp/p2" },
  ];
  store.selectedProjectId = "p1";
  return store;
}

async function send(wrapper: ReturnType<typeof mount>, text: string): Promise<void> {
  await wrapper.get('[data-testid="composer-input"]').setValue(text);
  await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
  await flushPromises();
}

beforeEach(() => {
  storage = createStorage();
  Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
  setActivePinia(createPinia());
});

afterEach(() => {
  if (originalStorage) Object.defineProperty(window, "localStorage", originalStorage);
  storage = null;
  newSessionProfile.value = null;
  vi.restoreAllMocks();
});

describe("HomeView Lifetime Invariants", () => {
  it.each(["resolve", "reject"])("ignores a late %s after auth reset", async (outcome) => {
    const store = prepareStore();
    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);
    const pushSpy = vi.spyOn(router, "push");
    const setItemSpy = vi.spyOn(storage!, "setItem");
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "private prompt");
    store.reset();
    await flushPromises();
    const writesAfterReset = setItemSpy.mock.calls.length;
    if (outcome === "resolve") deferred.resolve(sessionFixture);
    else deferred.reject(new Error("stale failure"));
    await flushPromises();
    expect(pushSpy).not.toHaveBeenCalled();
    expect(setItemSpy.mock.calls.length).toBe(writesAfterReset);
    expect(wrapper.find('[data-testid="home-error"]').exists()).toBe(false);
  });

  it("does not restore a draft after navigation rejects following unmount", async () => {
    const store = prepareStore();
    const navigation = createDeferred<undefined>();
    vi.spyOn(store, "createSession").mockResolvedValue(sessionFixture);
    vi.spyOn(router, "push").mockReturnValue(navigation.promise);
    const setItemSpy = vi.spyOn(storage!, "setItem");
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "private prompt");
    const writesBeforeUnmount = setItemSpy.mock.calls.length;
    wrapper.unmount();
    navigation.reject(new Error("navigation failed"));
    await flushPromises();
    expect(setItemSpy.mock.calls.length).toBe(writesBeforeUnmount);
  });

  it("restores the draft if current navigation rejects", async () => {
    const store = prepareStore();
    vi.spyOn(store, "createSession").mockResolvedValue(sessionFixture);
    vi.spyOn(router, "push").mockRejectedValue(new Error("navigation failed"));
    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "retry prompt");
    expect(wrapper.get('[data-testid="home-error"]').text()).toContain("navigation failed");
    expect(wrapper.get<HTMLTextAreaElement>('[data-testid="composer-input"]').element.value).toBe("retry prompt");
  });

  it("does not navigate, clear the draft or restore it when unmounted during a successful createSession", async () => {
    const store = prepareStore();
    const pushSpy = vi.spyOn(router, "push");
    const setItemSpy = vi.spyOn(storage!, "setItem");
    const removeItemSpy = vi.spyOn(storage!, "removeItem");

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "text");

    expect(store.createSession).toHaveBeenCalledTimes(1);
    const setItemCountBeforeUnmount = setItemSpy.mock.calls.length;
    const removeItemCountBeforeUnmount = removeItemSpy.mock.calls.length;

    wrapper.unmount();

    deferred.resolve(sessionFixture);
    await flushPromises();

    expect(pushSpy).not.toHaveBeenCalled();
    expect(setItemSpy.mock.calls.length).toBe(setItemCountBeforeUnmount);
    expect(removeItemSpy.mock.calls.length).toBe(removeItemCountBeforeUnmount);
  });

  it("does not restore the draft when createSession rejects after unmount", async () => {
    const store = prepareStore();
    const setItemSpy = vi.spyOn(storage!, "setItem");

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "important text");

    expect(store.createSession).toHaveBeenCalledTimes(1);
    const setItemCountBeforeUnmount = setItemSpy.mock.calls.length;

    wrapper.unmount();

    deferred.reject(new Error("API Error"));
    await flushPromises();

    expect(setItemSpy.mock.calls.length).toBe(setItemCountBeforeUnmount);
  });

  it("ignores a successful createSession that resolves after the project changed", async () => {
    const store = prepareStore();
    const pushSpy = vi.spyOn(router, "push");
    const setItemSpy = vi.spyOn(storage!, "setItem");

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "text");

    expect(store.createSession).toHaveBeenCalledTimes(1);
    const setItemCountBeforeChange = setItemSpy.mock.calls.length;

    store.selectedProjectId = "p2";
    await flushPromises();

    deferred.resolve(sessionFixture);
    await flushPromises();

    expect(pushSpy).not.toHaveBeenCalled();
    expect(setItemSpy.mock.calls.length).toBe(setItemCountBeforeChange);
  });

  it("ignores a failed createSession that rejects after the project changed", async () => {
    const store = prepareStore();
    const setItemSpy = vi.spyOn(storage!, "setItem");

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "important text");

    expect(store.createSession).toHaveBeenCalledTimes(1);
    const setItemCountBeforeChange = setItemSpy.mock.calls.length;

    store.selectedProjectId = "p2";
    await flushPromises();

    deferred.reject(new Error("API Error"));
    await flushPromises();

    expect(setItemSpy.mock.calls.length).toBe(setItemCountBeforeChange);
    expect(wrapper.find('[data-testid="home-error"]').exists()).toBe(false);
  });

  it("ignores a late response after switching A -> B -> A", async () => {
    const store = prepareStore();
    const pushSpy = vi.spyOn(router, "push");
    const setItemSpy = vi.spyOn(storage!, "setItem");

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "text");
    expect(store.createSession).toHaveBeenCalledTimes(1);
    const setItemCountBeforeChange = setItemSpy.mock.calls.length;

    store.selectedProjectId = "p2";
    store.selectedProjectId = "p1";
    await flushPromises();

    deferred.resolve(sessionFixture);
    await flushPromises();

    expect(pushSpy).not.toHaveBeenCalled();
    expect(setItemSpy.mock.calls.length).toBe(setItemCountBeforeChange);
  });

  it("old request finally does not clear the isCreating flag of a new pending request", async () => {
    const store = prepareStore();
    const firstDeferred = createDeferred<Session>();
    const secondDeferred = createDeferred<Session>();
    vi.spyOn(store, "createSession")
      .mockReturnValueOnce(firstDeferred.promise)
      .mockReturnValueOnce(secondDeferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "first");
    expect(store.createSession).toHaveBeenCalledTimes(1);

    store.selectedProjectId = "p2";
    await flushPromises();

    await send(wrapper, "second");
    expect(store.createSession).toHaveBeenCalledTimes(2);

    firstDeferred.resolve(sessionFixture);
    await flushPromises();
    expect(wrapper.get('[data-testid="composer-send"]').attributes("disabled")).toBeDefined();

    const pushSpy = vi.spyOn(router, "push").mockResolvedValue();
    secondDeferred.resolve(sessionFixture);
    await flushPromises();
    expect(pushSpy).toHaveBeenCalledWith("/s/s-123");
    await wrapper.get('[data-testid="composer-input"]').setValue("next prompt");
    expect(wrapper.get('[data-testid="composer-send"]').attributes("disabled")).toBeUndefined();
  });

  it("navigates on a successful createSession in the current context", async () => {
    const store = prepareStore();
    const pushSpy = vi.spyOn(router, "push").mockResolvedValue();

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "текст");

    deferred.resolve(sessionFixture);
    await flushPromises();

    expect(pushSpy).toHaveBeenCalledWith("/s/s-123");
    await wrapper.get('[data-testid="composer-input"]').setValue("next prompt");
    expect(wrapper.get('[data-testid="composer-send"]').attributes("disabled")).toBeUndefined();
  });

  it("restores the draft and shows the error when createSession fails in the current context", async () => {
    const store = prepareStore();

    const deferred = createDeferred<Session>();
    vi.spyOn(store, "createSession").mockReturnValue(deferred.promise);

    const wrapper = mount(HomeView, { global: { plugins: [router] } });
    await send(wrapper, "важный черновик");

    deferred.reject(new Error("boom"));
    await flushPromises();

    expect(wrapper.get('[data-testid="home-error"]').text()).toContain("boom");
    expect(
      (wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value,
    ).toBe("важный черновик");
    expect(wrapper.get('[data-testid="composer-send"]').attributes("disabled")).toBeUndefined();
  });
});

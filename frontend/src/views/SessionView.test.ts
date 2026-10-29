import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createRouter, createMemoryHistory } from "vue-router";
import SessionView from "./SessionView.vue";
import { useSessionStore } from "../stores/session";
import { useAppStore } from "../stores/app";
import type { Message } from "../api/types";

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
  routes: [{ path: "/s/:id", component: SessionView, props: true }],
});

function apiMessage(id: number, overrides: Partial<Message> = {}): Message {
  return {
    id,
    session_id: "s1",
    role: "user",
    content: "привет",
    tool_call_id: null,
    tool_calls: null,
    token_count: null,
    skipped: false,
    kind: "normal",
    reasoning_content: null,
    reasoning_effort: null,
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function prepareSession(id = "s1"): ReturnType<typeof useSessionStore> {
  const session = useSessionStore();
  session.sessionId = id;
  session.session = {
    id,
    project_id: "p1",
    parent_id: null,
    mode: "dev",
    llm_profile: "default",
    title: "Тестовая сессия",
    status: "completed",
    error: null,
    archived: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
  session.messages = [
    apiMessage(1),
    apiMessage(2, { role: "assistant", kind: "assistant", content: "готово" }),
  ];
  return session;
}

beforeEach(() => {
  storage = createStorage();
  Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
  setActivePinia(createPinia());
});

afterEach(() => {
  Object.defineProperty(window, "localStorage", { value: undefined, configurable: true });
  storage = null;
  vi.restoreAllMocks();
});

describe("SessionView", () => {
  it("retains a long session title in its tooltip", async () => {
    const store = prepareSession();
    const title = "Очень длинный заголовок сессии ".repeat(30);
    store.session!.title = title;
    const wrapper = mount(SessionView, { props: { id: "s1" }, global: { plugins: [router] } });
    await flushPromises();
    expect(wrapper.get('[data-testid="session-title"]').text()).toBe(title.trim());
    expect(wrapper.get('[data-testid="session-title"]').attributes("title")).toBe(title);
  });

  it("shows a localized status while retaining the server status in data", async () => {
    prepareSession();
    const wrapper = mount(SessionView, { props: { id: "s1" }, global: { plugins: [router] } });
    await flushPromises();
    const status = wrapper.get('[data-testid="session-status"]');
    expect(status.text()).toBe("Завершена");
    expect(status.attributes("data-status")).toBe("completed");
  });

  it("renders the session title and timeline messages", async () => {
    prepareSession();
    useAppStore().config = null;
    const wrapper = mount(SessionView, {
      props: { id: "s1" },
      global: { plugins: [router] },
    });
    await flushPromises();

    expect(wrapper.get('[data-testid="session-title"]').text()).toBe("Тестовая сессия");
    expect(wrapper.find('[data-testid="chat-timeline"]').exists()).toBe(true);
  });

  it("opens the session in the store on mount", async () => {
    const sessionStore = useSessionStore();
    sessionStore.sessionId = null;
    sessionStore.session = null;
    const openSpy = vi
      .spyOn(sessionStore, "open")
      .mockResolvedValue(undefined);

    const wrapper = mount(SessionView, {
      props: { id: "stale" },
      global: { plugins: [router] },
    });
    await flushPromises();
    expect(openSpy).toHaveBeenCalledWith("stale");
    wrapper.unmount();
  });

  it("sends a message through the store and keeps the draft per session", async () => {
    prepareSession();
    const sendSpy = vi.spyOn(useSessionStore(), "send").mockResolvedValue(undefined);
    const wrapper = mount(SessionView, {
      props: { id: "s1" },
      global: { plugins: [router] },
    });
    await flushPromises();

    await wrapper.get('[data-testid="composer-input"]').setValue("продолжим");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    await flushPromises();

    expect(sendSpy).toHaveBeenCalledWith("продолжим", undefined);
  });

  it("calls stop when the Composer emits stop", async () => {
    prepareSession();
    const stopSpy = vi.spyOn(useSessionStore(), "stop").mockResolvedValue(undefined);
    const wrapper = mount(SessionView, {
      props: { id: "s1" },
      global: { plugins: [router] },
    });
    await flushPromises();
    // Force busy state to show the Stop button.
    const store = useSessionStore();
    if (store.session) store.session = { ...store.session, status: "running" };
    await flushPromises();

    const stopBtn = wrapper.find('[data-testid="composer-stop"]');
    expect(stopBtn.exists()).toBe(true);
    await stopBtn.trigger("click");
    expect(stopSpy).toHaveBeenCalled();
  });

  it("shows an empty state when there are no messages", async () => {
    const store = prepareSession();
    store.messages = [];
    const wrapper = mount(SessionView, {
      props: { id: "s1" },
      global: { plugins: [router] },
    });
    await flushPromises();
    expect(wrapper.get('[data-testid="session-empty"]').text()).toContain("Сообщений пока нет");
  });

  it("shows openError for a failed open of B instead of stale data from A", async () => {
    prepareSession("s1"); // store is loaded with session A
    const sessionStore = useSessionStore();
    const openSpy = vi
      .spyOn(sessionStore, "open")
      .mockRejectedValue(new Error("cannot open B"));
    const wrapper = mount(SessionView, {
      props: { id: "s2" },
      global: { plugins: [router] },
    });
    await flushPromises();

    expect(openSpy).toHaveBeenCalledWith("s2");
    // A's content must not leak into the B route.
    expect(wrapper.find('[data-testid="session-title"]').exists()).toBe(false);
    expect(wrapper.get('[data-testid="open-error"]').text()).toContain("cannot open B");
  });

  it("retries opening B after an error and then renders B", async () => {
    const sessionStore = useSessionStore();
    prepareSession("s1"); // A is loaded
    const openSpy = vi
      .spyOn(sessionStore, "open")
      .mockRejectedValueOnce(new Error("first attempt failed"))
      .mockImplementation(async (id: string) => {
        sessionStore.sessionId = id;
        sessionStore.session = {
          id,
          project_id: "p1",
          parent_id: null,
          mode: "dev",
          llm_profile: "default",
          title: "Сессия B",
          status: "completed",
          error: null,
          archived: false,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        };
        sessionStore.messages = [];
      });

    const wrapper = mount(SessionView, {
      props: { id: "s2" },
      global: { plugins: [router] },
    });
    await flushPromises();
    expect(wrapper.get('[data-testid="open-error"]').text()).toContain("first attempt failed");

    await wrapper.get('[data-testid="open-retry"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-testid="session-title"]').text()).toBe("Сессия B");
    expect(wrapper.find('[data-testid="open-error"]').exists()).toBe(false);
    expect(openSpy).toHaveBeenCalledTimes(2);
  });

  it("restores the draft when sending a message fails", async () => {
    prepareSession();
    const sendSpy = vi
      .spyOn(useSessionStore(), "send")
      .mockRejectedValue(new Error("send failed"));
    const wrapper = mount(SessionView, {
      props: { id: "s1" },
      global: { plugins: [router] },
    });
    await flushPromises();

    await wrapper.get('[data-testid="composer-input"]').setValue("не потерять текст");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    await flushPromises();

    expect(sendSpy).toHaveBeenCalledWith("не потерять текст", undefined);
    expect(wrapper.get('[data-testid="send-error"]').text()).toContain("send failed");
    // The lost text is restored into the composer.
    expect(
      (wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value,
    ).toBe("не потерять текст");
  });
});

it("shows engine usage and unavailable health without history mutations or tuning", async () => {
  const store = prepareSession();
  Object.assign(store.session!, { llm_profile: null, engine_profile: "personal", engine_state: {
    health: { status: "unavailable", diagnostic: "engine недоступен" },
    usage: { input_tokens: 12, output_tokens: 3, cache_tokens: null, reported_cost: null, estimated_cost: true },
  } });
  const wrapper = mount(SessionView, { props: { id: "s1" }, global: { plugins: [router] } });
  await flushPromises();
  expect(wrapper.get('[data-testid="engine-health"]').text()).toContain("недоступен");
  expect(wrapper.get('[data-testid="engine-usage"]').text()).toContain("12");
  expect(wrapper.get('[data-testid="engine-usage"]').text()).toContain("неизвестна");
  expect(wrapper.find('[data-testid="composer-effort"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="edit-message"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="regenerate"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="fork"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="switch-branch"]').exists()).toBe(false);
  wrapper.unmount();
});

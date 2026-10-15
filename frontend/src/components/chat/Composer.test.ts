import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Composer from "./Composer.vue";

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

function mountComposer(props: Record<string, unknown> = {}) {
  return mount(Composer, {
    props: {
      busy: false,
      draftKey: "draft:test",
      efforts: ["auto", "high"],
      placeholder: "Что делаем?",
      ...props,
    },
  });
}

beforeEach(() => {
  storage = createStorage();
  Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
});

afterEach(() => {
  Object.defineProperty(window, "localStorage", { value: undefined, configurable: true });
  storage = null;
  vi.restoreAllMocks();
});

describe("Composer", () => {
  it("Enter sends the text and the selected effort, then clears the input", async () => {
    const wrapper = mountComposer();
    await wrapper.get('[data-testid="composer-input"]').setValue("найти баг");
    await wrapper.get('[data-testid="composer-effort"]').setValue("high");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("send")).toBeTruthy();
    expect(wrapper.emitted("send")?.[0]).toEqual([{ text: "найти баг", effort: "high" }]);
    expect((wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value).toBe("");
  });

  it("Shift+Enter inserts a newline and does not send", async () => {
    const wrapper = mountComposer();
    const input = wrapper.get('[data-testid="composer-input"]');
    await input.setValue("первая");
    await input.trigger("keydown", { key: "Enter", shiftKey: true });
    expect(wrapper.emitted("send")).toBeUndefined();
  });

  it("does not send empty or whitespace-only text", async () => {
    const wrapper = mountComposer();
    await wrapper.get('[data-testid="composer-input"]').setValue("   ");
    await wrapper.get('[data-testid="composer-input"]').trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("send")).toBeUndefined();
  });

  it("rapid double Enter emits send only once (input is cleared after send)", async () => {
    const wrapper = mountComposer();
    const input = wrapper.get('[data-testid="composer-input"]');
    await input.setValue("текст");
    await input.trigger("keydown", { key: "Enter" });
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("send")).toHaveLength(1);
  });

  it("shows the Stop button and emits stop while busy, disabling input", async () => {
    const wrapper = mountComposer({ busy: true });
    expect(wrapper.find('[data-testid="composer-send"]').exists()).toBe(false);
    const stop = wrapper.get('[data-testid="composer-stop"]');
    expect(
      (wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).disabled,
    ).toBe(true);
    await stop.trigger("click");
    expect(wrapper.emitted("stop")).toBeTruthy();
  });

  it("loads a draft from localStorage by draftKey on mount", () => {
    storage?.setItem("draft:test", "сохранённый черновик");
    const wrapper = mountComposer();
    expect((wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value).toBe(
      "сохранённый черновик",
    );
  });

  it("persists typed text to localStorage", async () => {
    const wrapper = mountComposer();
    await wrapper.get('[data-testid="composer-input"]').setValue("поработаем");
    expect(storage?.getItem("draft:test")).toBe("поработаем");
  });

  it("a new draftKey loads its own draft", async () => {
    storage?.setItem("draft:2", "вторая сессия");
    const wrapper = mountComposer();
    await wrapper.setProps({ draftKey: "draft:2" });
    expect((wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value).toBe(
      "вторая сессия",
    );
  });

  it("ignores localStorage errors when reading a draft", () => {
    storage!.getItem = () => {
      throw new Error("denied");
    };
    expect(() => mountComposer()).not.toThrow();
    const wrapper = mountComposer();
    expect((wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement).value).toBe("");
  });

  it("ignores localStorage errors when writing a draft", async () => {
    storage!.setItem = () => {
      throw new Error("quota exceeded");
    };
    const wrapper = mountComposer();
    await expect(wrapper.get('[data-testid="composer-input"]').setValue("текст")).resolves.not.toThrow();
  });

  it("does not persist drafts when draftKey is empty", async () => {
    const setItem = vi.fn();
    storage!.setItem = setItem;
    const wrapper = mountComposer({ draftKey: "" });
    await wrapper.get('[data-testid="composer-input"]').setValue("без черновика");
    expect(setItem).not.toHaveBeenCalled();
  });

  it("defaults the effort to the first available level", () => {
    const wrapper = mountComposer({ efforts: ["low", "auto"] });
    expect((wrapper.get('[data-testid="composer-effort"]').element as HTMLSelectElement).value).toBe("low");
  });

  it("shows a disabled send action when busy work cannot be stopped", async () => {
    const wrapper = mountComposer({ busy: true, stoppable: false });
    expect(wrapper.find('[data-testid="composer-stop"]').exists()).toBe(false);
    expect(wrapper.get('[data-testid="composer-send"]').attributes("disabled")).toBeDefined();
    await wrapper.get('[data-testid="composer-send"]').trigger("click");
    expect(wrapper.emitted("send")).toBeUndefined();
  });

  it("moves the same controls to a host without losing the effort and removes them on unmount", async () => {
    const host = document.createElement("div");
    host.id = "test-controls-host";
    document.body.appendChild(host);
    const wrapper = mountComposer({ controlsTarget: "#test-controls-host" });
    try {
      await flushPromises();
      expect(wrapper.find('[data-testid="agent-controls"]').exists()).toBe(false);
      expect(host.querySelectorAll('[data-testid="agent-controls"]')).toHaveLength(1);
      const select = host.querySelector("select")!;
      select.value = "high";
      select.dispatchEvent(new Event("change", { bubbles: true }));
      await flushPromises();
      await wrapper.setProps({ controlsTarget: "#missing-host" });
      expect(host.childElementCount).toBe(0);
      expect((wrapper.get('[data-testid="composer-effort"]').element as HTMLSelectElement).value).toBe("high");
      await wrapper.setProps({ controlsTarget: "#test-controls-host" });
      expect(host.querySelectorAll('[data-testid="agent-controls"]')).toHaveLength(1);
      expect(host.querySelector("select")?.value).toBe("high");
    } finally {
      wrapper.unmount();
      expect(host.childElementCount).toBe(0);
      host.remove();
    }
  });

  it("grows the textarea and enables its own scrollbar when the content is tall", async () => {
    const wrapper = mountComposer();
    const input = wrapper.get('[data-testid="composer-input"]').element as HTMLTextAreaElement;
    Object.defineProperty(input, "scrollHeight", { value: 500, configurable: true });
    await input.dispatchEvent(new Event("input"));
    expect(input.style.height).toMatch(/px$/);
    expect(input.style.overflowY).toBe("auto");
  });
});

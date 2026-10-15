import { afterEach, describe, expect, it } from "vitest";
import { mount, type VueWrapper } from "@vue/test-utils";
import Topbar from "./Topbar.vue";

const mountedTopbars: VueWrapper[] = [];

function mountTopbar(props: Record<string, unknown> = {}) {
  const wrapper = mount(Topbar, {
    attachTo: document.body,
    props: {
      connection: "live",
      ...props,
    },
  });
  mountedTopbars.push(wrapper);
  return wrapper;
}

async function dispatchKey(init: KeyboardEventInit): Promise<void> {
  const event = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init });
  document.body.dispatchEvent(event);
}

afterEach(() => {
  for (const wrapper of mountedTopbars.splice(0)) wrapper.unmount();
  document.body.innerHTML = "";
});

describe("Topbar", () => {
  it("shows the brand and a search placeholder with the ⌘K hint", () => {
    const wrapper = mountTopbar();
    expect(wrapper.find(".topbar__brand").text()).toContain("Агент");
    expect(wrapper.find('[data-search]').text()).toContain("Поиск");
    expect(wrapper.find(".topbar__kbd").text()).toBe("⌘K");
  });

  it("renders the connection indicator for each state", () => {
    const live = mountTopbar({ connection: "live" });
    expect(live.find('[data-connection="live"]').text()).toContain("Live");

    const offline = mountTopbar({ connection: "offline" });
    expect(offline.find('[data-connection="offline"]').text()).toContain("Нет связи");

    const connecting = mountTopbar({ connection: "connecting" });
    expect(connecting.find('[data-connection="connecting"]').text()).toContain("Подключение…");
  });

  it("emits search when the search field is clicked", async () => {
    const wrapper = mountTopbar();
    await wrapper.find('[data-search]').trigger("click");
    expect(wrapper.emitted("search")).toHaveLength(1);
  });

  it("emits search on ⌘K outside text fields", async () => {
    const wrapper = mountTopbar();
    await dispatchKey({ key: "k", metaKey: true });
    expect(wrapper.emitted("search")).toHaveLength(1);
  });

  it("emits search on Ctrl+K too", async () => {
    const wrapper = mountTopbar();
    await dispatchKey({ key: "k", ctrlKey: true });
    expect(wrapper.emitted("search")).toHaveLength(1);
  });

  it("does not emit search for ⌘K inside a text field", async () => {
    const wrapper = mountTopbar();
    const input = document.createElement("input");
    document.body.appendChild(input);

    const event = new KeyboardEvent("keydown", {
      key: "k",
      metaKey: true,
      bubbles: true,
      cancelable: true,
    });
    input.dispatchEvent(event);
    expect(wrapper.emitted("search")).toBeUndefined();
  });

  it("ignores unrelated keys", async () => {
    const wrapper = mountTopbar();
    await dispatchKey({ key: "a", metaKey: true });
    await dispatchKey({ key: "k" });
    expect(wrapper.emitted("search")).toBeUndefined();
  });

  it("removes the global ⌘K listener on unmount (no leaked handler)", async () => {
    const wrapper = mountTopbar();
    wrapper.unmount();
    await dispatchKey({ key: "k", metaKey: true });
    expect(wrapper.emitted("search")).toBeUndefined();
  });

  it("does not duplicate the composer model picker", () => {
    const wrapper = mountTopbar();
    expect(wrapper.find("select").exists()).toBe(false);
  });

  it("emits toggleTheme when the theme button is clicked", async () => {
    const wrapper = mountTopbar();
    await wrapper.find('[data-theme-toggle]').trigger("click");
    expect(wrapper.emitted("toggleTheme")).toHaveLength(1);
  });
});

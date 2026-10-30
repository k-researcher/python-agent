import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const tokenize = vi.fn();
vi.mock("../../highlight", async (original) => ({
  ...(await original<typeof import("../../highlight")>()),
  loadTokenizer: () => Promise.resolve(tokenize),
}));

import CodeView from "./CodeView.vue";

beforeEach(() => {
  tokenize.mockReset();
});

describe("CodeView", () => {
  it("shows plain lines at once, with diff markers and line kinds", () => {
    tokenize.mockReturnValue(new Promise(() => {}));
    const wrapper = mount(CodeView, {
      props: {
        lang: "python",
        lines: [
          { text: "keep()", type: "context" },
          { text: "old()", type: "remove" },
          { text: "new()", type: "add" },
        ],
      },
    });
    const rows = wrapper.findAll('[data-testid="code-line"]');
    expect(rows.map((row) => row.attributes("data-type"))).toEqual(["context", "remove", "add"]);
    expect(rows.map((row) => row.text())).toEqual(["keep()", "−old()", "+new()"]);
  });

  it("colours the code after the highlighter loads, as one block for multi-line syntax", async () => {
    tokenize.mockResolvedValue([
      [{ content: "def", color: "#ff7b72" }, { content: " f():", color: "#e6edf3" }],
      [{ content: "    pass", color: "#e6edf3" }],
    ]);
    const wrapper = mount(CodeView, {
      props: { lang: "python", lines: [{ text: "def f():", type: "add" }, { text: "    pass", type: "add" }] },
    });
    await flushPromises();
    expect(tokenize).toHaveBeenCalledWith("def f():\n    pass", "python");
    const first = wrapper.findAll('[data-testid="code-line"]')[0];
    expect(first.find('span[style*="color"]').text()).toBe("def");
    expect(first.text()).toBe("+def f():");
  });

  it("colours the old and the new side of a diff separately", async () => {
    tokenize.mockImplementation(async (code: string) =>
      code.split("\n").map((line) => [{ content: line, color: code.includes("old") ? "#f00" : "#0f0" }]),
    );
    const wrapper = mount(CodeView, {
      props: {
        lang: "python",
        lines: [
          { text: "same", type: "context" },
          { text: "old", type: "remove" },
          { text: "new", type: "add" },
        ],
      },
    });
    await flushPromises();
    expect(tokenize).toHaveBeenCalledWith("same\nold", "python");
    expect(tokenize).toHaveBeenCalledWith("same\nnew", "python");
    const colours = wrapper
      .findAll('[data-testid="code-line"]')
      .map((row) => row.find("span[style]").attributes("style"));
    expect(colours[1]).toContain("#f00");
    expect(colours[2]).toContain("#0f0");
    expect(colours[0]).toContain("#0f0");
  });

  it("keeps plain text when the language is unknown or the highlighter fails", async () => {
    tokenize.mockRejectedValue(new Error("no grammar"));
    const failed = mount(CodeView, { props: { lang: "python", lines: [{ text: "x = 1" }] } });
    await flushPromises();
    expect(failed.get('[data-testid="code-line"]').text()).toBe("x = 1");

    const plain = mount(CodeView, { props: { lang: null, lines: [{ text: "x = 1" }] } });
    await flushPromises();
    expect(plain.get('[data-testid="code-line"]').text()).toBe("x = 1");
    expect(tokenize).toHaveBeenCalledTimes(1);
  });

  it("does not colour a very long block", async () => {
    const lines = Array.from({ length: 5000 }, (_, i) => ({ text: `line ${i}` }));
    mount(CodeView, { props: { lang: "python", lines } });
    await flushPromises();
    expect(tokenize).not.toHaveBeenCalled();
  });

  it("numbers the lines from the given start", () => {
    tokenize.mockReturnValue(new Promise(() => {}));
    const wrapper = mount(CodeView, {
      props: { lang: null, diff: false, start: 41, lines: [{ text: "a" }, { text: "b" }] },
    });
    expect(wrapper.findAll('[data-testid="code-number"]').map((n) => n.text())).toEqual(["41", "42"]);
  });

  it("wraps long lines by default and keeps them on one line when wrap is off", async () => {
    tokenize.mockReturnValue(new Promise(() => {}));
    const wrapper = mount(CodeView, { props: { lines: [{ text: "x".repeat(400) }], diff: false } });
    const view = wrapper.get('[data-testid="code-view"]');
    expect(view.classes()).not.toContain("code-view--nowrap");
    await wrapper.get('[data-testid="code-wrap"]').trigger("click");
    expect(view.classes()).toContain("code-view--nowrap");
  });

  it("copies the code after the change from the button inside the block", async () => {
    tokenize.mockReturnValue(new Promise(() => {}));
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    const wrapper = mount(CodeView, {
      props: { lines: [{ text: "a = 1", type: "remove" }, { text: "a = 2", type: "add" }] },
    });
    const button = wrapper.get('[data-testid="code-copy"]');
    expect(wrapper.get(".code-frame").element.contains(button.element)).toBe(true);
    await button.trigger("click");
    await flushPromises();
    expect(writeText).toHaveBeenCalledWith("a = 2");
    expect(button.text()).toBe("Скопировано");
    vi.unstubAllGlobals();
  });
});

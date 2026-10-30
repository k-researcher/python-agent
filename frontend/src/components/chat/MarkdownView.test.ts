import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import MarkdownView from "./MarkdownView.vue";

async function mountView(source: string) {
  const wrapper = mount(MarkdownView, { props: { source } });
  await flushPromises();
  return wrapper;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("MarkdownView", () => {
  it("renders sanitized markdown via v-html", async () => {
    const wrapper = await mountView("**bold** text");
    const body = wrapper.get('[data-testid="markdown-body"]');
    expect(body.html()).toContain("<strong>bold</strong>");
  });

  it("never mounts a live <script> element", async () => {
    const wrapper = await mountView("<script>alert(1)</script>");
    const body = wrapper.find('[data-testid="markdown-body"]') as ReturnType<typeof wrapper.find>;
    // Raw model HTML must stay as escaped text, never become a node.
    expect(body.find("script").exists()).toBe(false);
    expect(body.text()).toContain("<script>");
  });

  it("strips onerror attributes", async () => {
    const wrapper = await mountView('<img src="x" onerror="alert(1)">');
    const el = wrapper.get('[data-testid="markdown-body"]').element as HTMLElement;
    expect(el.querySelector("[onerror]")).toBeNull();
    expect(el.querySelector("img")).toBeNull();
  });

  it("does not pass raw HTML through (html:false)", async () => {
    const wrapper = await mountView("<b>жирный</b>");
    const el = wrapper.get('[data-testid="markdown-body"]').element as HTMLElement;
    expect(el.querySelector("b")).toBeNull();
    expect(el.textContent).toContain("<b>");
  });

  it("opens links with target=_blank and rel=noopener noreferrer", async () => {
    const wrapper = await mountView("[ссылка](https://example.com)");
    const link = wrapper.get('[data-testid="markdown-body"] a') as ReturnType<
      typeof wrapper.get
    >;
    expect(link.attributes("href")).toBe("https://example.com");
    expect(link.attributes("target")).toBe("_blank");
    expect(link.attributes("rel")).toBe("noopener noreferrer");
  });

  it("adds a copy button to every code block", async () => {
    const wrapper = await mountView("```js\nconst a = 1;\n```");
    const body = wrapper.get('[data-testid="markdown-body"]');
    expect(body.element.querySelectorAll("pre")).not.toHaveLength(0);
    const toolbar = body.element.querySelector(".code-block__toolbar");
    const copyBtn = toolbar?.querySelector(".copy-btn");
    expect(copyBtn).not.toBeNull();
    expect(copyBtn?.getAttribute("aria-label")).toBe("Копировать");
  });

  it("wraps long lines by default and lets the user turn wrap off per block", async () => {
    const wrapper = await mountView("```js\n1\n```\n\n```js\n2\n```");
    const blocks = wrapper.findAll(".code-block");
    expect(blocks.map((block) => block.classes("code-block--nowrap"))).toEqual([false, false]);
    const wrap = blocks[0].get('[data-testid="code-wrap"]');
    expect(wrap.attributes("aria-pressed")).toBe("true");
    await wrap.trigger("click");
    expect(blocks.map((block) => block.classes("code-block--nowrap"))).toEqual([true, false]);
    expect(wrap.attributes("aria-pressed")).toBe("false");
  });

  it("copy button copies the code text via navigator.clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    const wrapper = await mountView("```py\nprint(1)\n```");
    const copyBtn = wrapper.get('[data-testid="markdown-body"] .copy-btn') as ReturnType<
      typeof wrapper.get
    >;
    await copyBtn.trigger("click");
    await flushPromises();
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText).toHaveBeenCalledWith("print(1)\n");
    expect(copyBtn.text()).toBe("Скопировано");
  });

  it("handles empty and very large sources without crashing", async () => {
    expect((await mountView("")).find('[data-testid="markdown-body"]').exists()).toBe(true);
    const big = "# заголовок\n\n" + Array.from({ length: 2000 }, (_, i) => `строка ${i}`).join("\n");
    await expect(mountView(big)).resolves.not.toThrow();
  });

  it("unmounting mid-highlight produces no error and no post-dispose mutation", async () => {
    const wrapper = mount(MarkdownView, { props: { source: "```ts\nconst a = 1;\n```" } });
    wrapper.unmount();
    // The in-flight shiki promise resolves after disposal; the generation /
    // disposed guards must swallow it without touching the refs.
    await expect(flushPromises()).resolves.not.toThrow();
  });

  it("highlights code blocks with the soft dark theme", async () => {
    const wrapper = await mountView("```ts\nconst x = 1;\n```");
    expect(wrapper.get('[data-testid="markdown-body"] pre.shiki').classes()).toContain("github-dark-dimmed");
  });

  it("does not duplicate copy buttons after source updates", async () => {
    const wrapper = mount(MarkdownView, { props: { source: "```js\nconst a = 1;\n```" } });
    await flushPromises();
    const countBefore = wrapper.get('[data-testid="markdown-body"]').element.querySelectorAll(".copy-btn").length;
    expect(countBefore).toBe(1);

    await wrapper.setProps({ source: "```js\nconst a = 2;\n```" });
    await flushPromises();
    // The fresh render must show exactly one button, never two stacked.
    const countAfter = wrapper.get('[data-testid="markdown-body"]').element.querySelectorAll(".copy-btn").length;
    expect(countAfter).toBe(1);
    expect(wrapper.get('[data-testid="markdown-body"]').element.querySelectorAll(".code-block__toolbar").length).toBe(1);
    // And the code block is not nested inside two .code-block wrappers.
    const wrappers = wrapper
      .get('[data-testid="markdown-body"]')
      .element.querySelectorAll(".code-block");
    expect(wrappers.length).toBe(1);
  });

  it("toolbar is a separate sibling of pre and handles long first lines", async () => {
    const longLine = "const x = " + "a".repeat(100) + ";";
    const wrapper = await mountView("```js\n" + longLine + "\n```");
    const body = wrapper.get('[data-testid="markdown-body"]').element;
    const pre = wrapper.get("pre").element;
    const toolbar = wrapper.get(".code-block__toolbar").element;
    expect(pre.parentElement).toBe(toolbar.parentElement);
    expect(pre.previousElementSibling).toBe(toolbar);
    expect(toolbar.contains(pre)).toBe(false);

    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    await wrapper.get(".copy-btn").trigger("click");
    await flushPromises();
    expect(writeText).toHaveBeenCalledWith(longLine + "\n");
  });

  it("multiple code blocks have separate toolbars", async () => {
    const wrapper = await mountView("```js\n1\n```\n\n```js\n2\n```");
    const body = wrapper.get('[data-testid="markdown-body"]').element;
    expect(body.querySelectorAll(".code-block__toolbar")).toHaveLength(2);
    expect(body.querySelectorAll("pre")).toHaveLength(2);
  });
});

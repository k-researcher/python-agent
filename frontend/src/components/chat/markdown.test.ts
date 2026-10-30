import { describe, expect, it } from "vitest";
import { extractLanguage, highlightCodeBlocks, renderMarkdown } from "./markdown";

describe("renderMarkdown", () => {
  it("never emits a live <script> element", () => {
    const html = renderMarkdown("<script>alert(1)</script>");
    expect(html).not.toMatch(/<script/i);
    // The payload is escaped text, never parsed as markup.
    expect(html).toContain("&lt;script&gt;");
  });

  it("strips event handler attributes from the model payload", () => {
    const html = renderMarkdown('<img src="x" onerror="alert(1)">');
    // Whatever the serializer does, the payload must never mount a live
    // element with an event-handler attribute.
    const container = document.createElement("div");
    container.innerHTML = html;
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("[onerror]")).toBeNull();
  });

  it("does not output raw HTML elements (html:false)", () => {
    const html = renderMarkdown("<b>жирный</b>");
    expect(html).not.toMatch(/<b>/i);
  });

  it("blocks javascript: URLs from becoming links", () => {
    const html = renderMarkdown("[click](javascript:alert(1))");
    expect(html).not.toMatch(/href\s*=\s*["']?javascript:/i);
  });

  it("adds target=_blank and rel=noopener noreferrer to links", () => {
    const html = renderMarkdown("[ссылка](https://example.com)");
    expect(html).toContain('href="https://example.com"');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
  });

  it("keeps the anchor text when a link destination is blocked", () => {
    const html = renderMarkdown("[click](javascript:alert(1))");
    expect(html).toContain("click");
  });

  it("handles empty, blank and oversized inputs without throwing", () => {
    expect(renderMarkdown("")).not.toBeNull();
    expect(renderMarkdown(undefined as unknown as string)).not.toBeNull();
    expect(renderMarkdown("   ")).not.toBeNull();
    expect(renderMarkdown("a".repeat(50_000)).length).toBeGreaterThan(0);
  });

  it("keeps code fence content and language in the output", () => {
    const html = renderMarkdown("```python\nprint(1)\n```");
    expect(html).toContain("language-python");
    expect(html).toContain("print(1)");
  });
});

describe("extractLanguage", () => {
  function codeWithClass(className: string): Element {
    const el = document.createElement("code");
    el.setAttribute("class", className);
    return el;
  }

  it("reads a language from the class list", () => {
    expect(extractLanguage(codeWithClass("language-python"))).toBe("python");
    expect(extractLanguage(codeWithClass("hljs language-js"))).toBe("js");
  });

  it("falls back to text without a language class", () => {
    expect(extractLanguage(codeWithClass("hljs"))).toBe("text");
    expect(extractLanguage(codeWithClass(""))).toBe("text");
  });
});

describe("highlightCodeBlocks", () => {
  it("returns html unchanged when it has no code blocks", async () => {
    const input = "<p>без кода</p>";
    await expect(highlightCodeBlocks(input)).resolves.toBe(input);
  });

  it("highlights a known language and keeps the source readable", async () => {
    const input = "<pre><code class=\"language-js\">const x = 1</code></pre>";
    const out = await highlightCodeBlocks(input);
    expect(out).toContain("shiki");
    expect(out).toContain("const");
  });

  it("wraps a bare <code> block in <pre> before highlighting", async () => {
    // DOMPurify-based environs may drop the <pre> wrapper; normalization
    // must restore it so the block is still highlighted.
    const input = "<code class=\"language-js\">let a = 1</code>";
    const out = await highlightCodeBlocks(input);
    const container = document.createElement("div");
    container.innerHTML = out;
    expect(container.querySelector("pre.shiki")).not.toBeNull();
    expect(container.querySelector("pre.shiki")?.textContent).toBe("let a = 1");
  });

  it("keeps a plain block when the language cannot be highlighted", async () => {
    const input = "<pre><code class=\"language-definitely-not-a-real-lang\">tmp</code></pre>";
    const out = await highlightCodeBlocks(input);
    expect(out).toContain("tmp");
  });

  it("uses the same soft dark theme as the tool cards", async () => {
    const input = "<pre><code class=\"language-js\">const y = 2</code></pre>";
    const out = await highlightCodeBlocks(input);
    expect(out).toContain("github-dark-dimmed");
    expect(out).not.toContain("github-light");
    // The app tokens set the background, so the theme's inline colours are dropped.
    expect(out).not.toContain("background-color");
  });
});

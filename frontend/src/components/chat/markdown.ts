// Markdown rendering pipeline for chat messages:
// markdown-it (html:false, linkify:true) -> DOMPurify -> optional shiki highlighting.
//
// The secure-rendering guarantees (no raw model HTML, no event-handler
// attributes, safe link protocols) are provided by markdown-it's options and
// validateLink plus DOMPurify. Code blocks are normalized with plain DOM
// helpers afterwards, so a code fence always renders as <pre><code> even in
// environments whose sanitizer drops the wrapper element.

import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";

let parser: MarkdownIt | null = null;

function getParser(): MarkdownIt {
  if (parser) return parser;
  const md = new MarkdownIt({ html: false, linkify: true });

  // External links must always open in a new, isolated tab.
  // Parameter types are inferred from the RenderRule signature, so we avoid
  // depending on the (historically split) markdown-it namespace types.
  const defaultLinkOpen =
    md.renderer.rules.link_open ??
    ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));

  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    tokens[idx].attrSet("target", "_blank");
    tokens[idx].attrSet("rel", "noopener noreferrer");
    return defaultLinkOpen(tokens, idx, options, env, self);
  };

  parser = md;
  return parser;
}

/**
 * Render markdown to sanitized HTML. Never returns raw model HTML: raw tags
 * are escaped by markdown-it (html:false) and javascript:/data: URLs are
 * rejected by validateLink before the result reaches DOMPurify.
 */
export function renderMarkdown(source: string): string {
  const html = getParser().render(String(source ?? ""));
  return DOMPurify.sanitize(html);
}

interface HighlightApi {
  codeToHtml: (
    code: string,
    options: { lang: string; themes: { light: string; dark: string } },
  ) => Promise<string>;
}

let highlightApiPromise: Promise<HighlightApi> | null = null;

/** Lazily load shiki once; used only for blocks that actually contain code. */
function loadHighlightApi(): Promise<HighlightApi> {
  highlightApiPromise ??= import("shiki").then((module) => ({
    codeToHtml: module.codeToHtml,
  }));
  return highlightApiPromise;
}

/** Extract the language name from a code element class (language-*). */
export function extractLanguage(codeElement: Element): string {
  const className = typeof codeElement.getAttribute === "function" ? codeElement.getAttribute("class") ?? "" : "";
  const match = /(?:^|\s)language-([A-Za-z0-9_+-]+)/.exec(className);
  return match ? match[1] : "text";
}

/**
 * Normalize code blocks in parsed HTML: make sure every <code class="language-*">
 * is wrapped in a <pre>. DOMPurify (in some environments) may have dropped the
 * <pre> wrapper; this restores the structure deterministically so highlighting
 * and the copy button always have an anchor.
 */
function normalizeCodeBlocks(doc: Document & { body: HTMLElement }): void {
  const blocks = Array.from(doc.querySelectorAll("code"));
  for (const code of blocks) {
    const classAttr = code.getAttribute("class") ?? "";
    if (!/(?:^|\s)language-/.test(classAttr)) continue;
    if (code.parentElement && code.parentElement.tagName === "PRE") continue;
    const pre = doc.createElement("pre");
    pre.appendChild(code);
    doc.body.appendChild(pre);
  }
}

/**
 * Wrap every bare <code class="language-*"> element in a <pre>. Some sanitizer
 * environments drop the original <pre> wrapper, so callers call this on the
 * sanitized HTML before attaching copy buttons or highlighting.
 */
export function wrapCodeBlocksInPre(html: string): string {
  if (typeof DOMParser === "undefined" || typeof document === "undefined") return html;
  const doc = new DOMParser().parseFromString(html, "text/html");
  normalizeCodeBlocks(doc);
  return doc.body.innerHTML;
}

/**
 * Replace each code <pre> inside `html` with a shiki-highlighted version.
 * Highlighting is best-effort: on any failure the plain block is kept as-is.
 *
 * `dark` chooses the active theme: light or dark. Shiki is told to emit both
 * themed styles (CSS variable fallbacks) but the result is tagged with the
 * active theme so callers/themes can apply the right colors; both themes are
 * always present in the markup so toggling works without re-highlighting.
 */
export async function highlightCodeBlocks(html: string, dark: boolean): Promise<string> {
  if (typeof DOMParser === "undefined" || typeof document === "undefined") return html;
  const doc = new DOMParser().parseFromString(html, "text/html");
  normalizeCodeBlocks(doc);
  const pres = Array.from(doc.querySelectorAll("pre"));
  if (pres.length === 0) return html;

  const { codeToHtml } = await loadHighlightApi();
  for (const pre of pres) {
    const code = pre.querySelector("code");
    if (!code) continue;
    const classAttr = code.getAttribute("class") ?? "";
    if (!/(?:^|\s)language-/.test(classAttr)) continue;
    const lang = extractLanguage(code);
    const raw = code.textContent ?? "";
    try {
      const themed = await codeToHtml(raw, {
        lang,
        themes: { light: "github-light", dark: "github-dark" },
      });
      const holder = doc.createElement("div");
      holder.innerHTML = themed;
      const replacement = holder.firstElementChild;
      if (replacement) {
        pre.replaceWith(replacement);
        // Record the active theme so CSS can switch the visible palette.
        replacement.setAttribute("data-active-theme", dark ? "dark" : "light");
      }
    } catch {
      // Keep the unhighlighted block on errors or unknown language.
    }
  }
  return doc.body.innerHTML;
}

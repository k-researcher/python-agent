// Syntax colours for code blocks: shiki tokens on a soft dark theme, loaded lazily.

export interface CodeToken {
  content: string;
  color?: string;
}

export type Tokenize = (code: string, lang: string) => Promise<CodeToken[][]>;

/** Above this size the plain text is shown: colouring would cost more than it gives. */
export const MAX_HIGHLIGHT_LINES = 2000;

/** One soft dark theme for every code block: tool cards and chat answers. */
export const CODE_THEME = "github-dark-dimmed";

let tokenizer: Promise<Tokenize> | null = null;

export function loadTokenizer(): Promise<Tokenize> {
  tokenizer ??= import("shiki").then((shiki) => async (code: string, lang: string) => {
    const result = await shiki.codeToTokens(code, {
      lang: lang as Parameters<typeof shiki.codeToTokens>[1]["lang"],
      theme: CODE_THEME,
    });
    return result.tokens.map((line) => line.map((token) => ({ content: token.content, color: token.color })));
  });
  return tokenizer;
}

const EXTENSIONS: Record<string, string> = {
  py: "python",
  ts: "typescript",
  tsx: "tsx",
  js: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  jsx: "jsx",
  vue: "vue",
  json: "json",
  md: "markdown",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  ini: "ini",
  sh: "bash",
  bash: "bash",
  zsh: "bash",
  html: "html",
  css: "css",
  scss: "scss",
  sql: "sql",
  go: "go",
  rs: "rust",
  php: "php",
  rb: "ruby",
  java: "java",
  kt: "kotlin",
  swift: "swift",
  c: "c",
  h: "c",
  cpp: "cpp",
  cs: "csharp",
};

const NAMES: Record<string, string> = { dockerfile: "docker", makefile: "makefile" };

export function languageForPath(path: string | null | undefined): string | null {
  if (!path) return null;
  const name = path.split("/").pop()?.toLowerCase() ?? "";
  if (NAMES[name]) return NAMES[name];
  const dot = name.lastIndexOf(".");
  return dot > 0 ? (EXTENSIONS[name.slice(dot + 1)] ?? null) : null;
}

/** Command output gets colours only from its own ANSI escape codes. */
export function languageForOutput(text: string): string | null {
  return text.includes("\u001b[") ? "ansi" : null;
}

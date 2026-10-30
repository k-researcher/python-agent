// Pure helpers for ToolCallCard: brief-argument extraction, status labels,
// shell result shaping and old -> new file-edit diffs. Kept free of Vue so
// the logic can be unit-tested directly.

import type { ToolCallView } from "./types";
import { diffLines, type DiffLine } from "./diff";
import { languageForPath } from "../../highlight";

export const STATUS_LABEL: Record<string, string> = {
  pending: "Ожидание",
  awaiting_approval: "Требуется подтверждение",
  running: "Выполняется",
  completed: "Выполнено",
  error: "Ошибка",
  rejected: "Отклонено",
};

const FILE_TOOLS = new Set(["write_file", "edit_file", "multi_edit", "read_file", "read_many"]);
const PATTERN_TOOLS = new Set(["glob", "search_files"]);

export function statusLabel(status: string): string {
  return STATUS_LABEL[status] ?? status;
}

/**
 * Short one-line description of the call arguments:
 * `command` for shell, `path` for file tools, `pattern` for glob/search_files,
 * otherwise the first 60 characters of the JSON form.
 */
export function briefArgument(call: ToolCallView): string {
  const args = call.arguments ?? {};
  if (call.name === "shell") {
    const command = args["command"];
    if (typeof command === "string") return command;
  } else if (FILE_TOOLS.has(call.name)) {
    const path = args["path"];
    if (typeof path === "string") return path;
  } else if (PATTERN_TOOLS.has(call.name)) {
    const pattern = args["pattern"];
    if (typeof pattern === "string") return pattern;
  }
  return truncateJson(call);
}

function truncateJson(call: ToolCallView): string {
  const json = call.argumentsRaw && call.argumentsRaw.trim() ? call.argumentsRaw : safeStringify(call.arguments);
  const compact = (json ?? "").replace(/\s+/g, " ").trim();
  if (compact.length <= 60) return compact;
  return `${compact.slice(0, 60)}…`;
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return "";
  }
}

/** Human-readable JSON of the arguments (pretty-printed). */
export function prettyJson(call: ToolCallView): string {
  if (call.argumentsRaw && call.argumentsRaw.trim()) {
    try {
      return JSON.stringify(JSON.parse(call.argumentsRaw), null, 2);
    } catch {
      // fall through to the structured copy below
    }
  }
  return safeStringify(call.arguments);
}

export interface ShellOutput {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  error: string;
}

/** Extract stdout / stderr / exit_code from a shell call result, tolerantly. */
export function shellOutput(call: ToolCallView): ShellOutput {
  const result = call.result;
  const empty: ShellOutput = { stdout: "", stderr: "", exitCode: null, error: "" };
  if (!result) return empty;
  const error = typeof result.error === "string" ? result.error : "";
  const out = result.output;
  if (out && typeof out === "object" && !Array.isArray(out)) {
    const rec = out as Record<string, unknown>;
    return {
      stdout: typeof rec.stdout === "string" ? rec.stdout : "",
      stderr: typeof rec.stderr === "string" ? rec.stderr : "",
      exitCode: typeof rec.exit_code === "number" ? rec.exit_code : null,
      error,
    };
  }
  if (typeof out === "string") {
    return { stdout: out, stderr: "", exitCode: null, error };
  }
  return { stdout: "", stderr: "", exitCode: null, error };
}

export interface EditPair {
  path: string;
  oldText: string;
  newText: string;
}

/**
 * Extract old/new text pairs for write_file / edit_file / multi_edit.
 * multi_edit carries an `edits` array; `path` may live on the call or on
 * each edit entry.
 */
export function fileEdits(call: ToolCallView): EditPair[] {
  const args = call.arguments ?? {};
  const globalPath = typeof args["path"] === "string" ? (args["path"] as string) : "";
  const editsRaw = args["edits"];
  const pairs: EditPair[] = [];

  if (Array.isArray(editsRaw)) {
    for (const item of editsRaw) {
      const rec = objectFromUnknown(item);
      const content = typeof rec["content"] === "string" ? (rec["content"] as string) : null;
      pairs.push({
        path:
          typeof rec["path"] === "string" ? (rec["path"] as string) : globalPath,
        oldText: content === null ? (typeof rec["old_text"] === "string" ? (rec["old_text"] as string) : "") : "",
        newText:
          content !== null
            ? content
            : typeof rec["new_text"] === "string"
              ? (rec["new_text"] as string)
              : "",
      });
    }
  } else if (globalPath) {
    if (call.name === "write_file") {
      // write_file carries its payload in `content`, not old/new text.
      pairs.push({
        path: globalPath,
        oldText: "",
        newText: typeof args["content"] === "string" ? (args["content"] as string) : "",
      });
    } else {
      pairs.push({
        path: globalPath,
        oldText: typeof args["old_text"] === "string" ? (args["old_text"] as string) : "",
        newText: typeof args["new_text"] === "string" ? (args["new_text"] as string) : "",
      });
    }
  }
  return pairs;
}

/** Line diff for one edit pair (old -> new). */
export function editDiff(pair: EditPair): DiffLine[] {
  return diffLines(pair.oldText, pair.newText);
}

function objectFromUnknown(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string") {
    try {
      const parsed: unknown = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // bad JSON entry -> treat as empty
    }
  }
  return {};
}

export interface ResultBlock {
  label?: string;
  lines: { text: string }[];
  lang: string | null;
  /** Number of the first line, when the block is a part of a file. */
  start?: number;
}

function lines(text: string): { text: string }[] {
  return text.replace(/\n$/, "").split("\n").map((line) => ({ text: line }));
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function bodyLanguage(contentType: string): string | null {
  const type = contentType.toLowerCase();
  if (type.includes("json")) return "json";
  if (type.includes("html")) return "html";
  if (type.includes("xml")) return "xml";
  if (type.includes("markdown")) return "markdown";
  if (type.includes("javascript")) return "javascript";
  if (type.includes("css")) return "css";
  return null;
}

function formatted(body: string, lang: string | null): string {
  if (lang !== "json") return body;
  try {
    return JSON.stringify(JSON.parse(body), null, 2);
  } catch {
    return body;
  }
}

/**
 * The result of a finished tool call as code blocks, each in its own language.
 * Shell, file edits and engine patches have their own views in ToolCallCard.
 */
export function resultBlocks(call: ToolCallView): ResultBlock[] {
  const result = call.result;
  if (!result || !result.success) return [];
  const output = result.output;
  const out = record(output);
  const args = call.arguments ?? {};
  switch (call.name) {
    case "read_file":
      return [
        {
          lines: lines(text(out.content)),
          lang: languageForPath(text(out.path) || text(args.path)),
          start: typeof out.start_line === "number" ? out.start_line : 1,
        },
      ];
    case "list_dir": {
      const entries = Array.isArray(out.entries) ? out.entries.map(record) : [];
      const names = entries.map((entry) => text(entry.name) + (entry.type === "directory" ? "/" : ""));
      return [{ lines: names.length ? names.map((name) => ({ text: name })) : [{ text: "Каталог пуст" }], lang: null }];
    }
    case "glob": {
      const paths = Array.isArray(out.paths) ? out.paths.filter((p): p is string => typeof p === "string") : [];
      return [{ lines: paths.length ? paths.map((path) => ({ text: path })) : [{ text: "Ничего не найдено" }], lang: null }];
    }
    case "search_files": {
      const matches = Array.isArray(out.matches) ? out.matches.map(record) : [];
      const found = matches.map((m) => ({ text: `${text(m.path)}:${String(m.line ?? "")}  ${text(m.text)}` }));
      return [{ lines: found.length ? found : [{ text: "Ничего не найдено" }], lang: null }];
    }
    case "git_diff":
      return [{ lines: lines(text(out.diff)), lang: "diff" }];
    case "git_show":
      return [{ lines: lines(text(out.output)), lang: "diff" }];
    case "apply_patch":
      return [{ lines: lines(text(args.patch)), lang: "diff" }];
    case "web_fetch":
      return [{ label: text(out.url) || undefined, lines: lines(text(out.markdown)), lang: "markdown" }];
    case "http_request": {
      const lang = bodyLanguage(text(out.content_type));
      return [
        { lines: [{ text: `HTTP ${String(out.status ?? "")} ${text(out.content_type)}`.trim() }], lang: null },
        { lines: lines(formatted(text(out.body), lang)), lang },
      ];
    }
  }
  if (typeof output === "string") return [{ lines: lines(output), lang: null }];
  if (output == null) return [];
  return [{ lines: lines(JSON.stringify(output, null, 2)), lang: "json" }];
}

// Pure helpers for ToolCallCard: brief-argument extraction, status labels,
// shell result shaping and old -> new file-edit diffs. Kept free of Vue so
// the logic can be unit-tested directly.

import type { ToolCallView } from "./types";
import { diffLines, type DiffLine } from "./diff";

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

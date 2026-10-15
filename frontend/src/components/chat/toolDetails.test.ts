import { describe, expect, it } from "vitest";
import type { ToolCallView } from "./types";
import {
  briefArgument,
  editDiff,
  fileEdits,
  prettyJson,
  shellOutput,
  statusLabel,
} from "./toolDetails";

function makeCall(partial: Partial<ToolCallView>): ToolCallView {
  return {
    id: "c1",
    name: "shell",
    arguments: {},
    argumentsRaw: "",
    result: null,
    approval: null,
    status: "pending",
    ...partial,
  };
}

const longJson = JSON.stringify({
  command: "npm run build && npm test -- --watch --coverage",
  cwd: "/very/long/project/path/that/keeps/growing/even/longer",
});

describe("statusLabel", () => {
  it("maps known statuses to Russian labels", () => {
    expect(statusLabel("awaiting_approval")).toBe("Требуется подтверждение");
    expect(statusLabel("completed")).toBe("Выполнено");
    expect(statusLabel("error")).toBe("Ошибка");
    expect(statusLabel("rejected")).toBe("Отклонено");
  });

  it("passes unknown statuses through unchanged", () => {
    expect(statusLabel("weird_status")).toBe("weird_status");
  });
});

describe("briefArgument", () => {
  it("uses `command` for shell", () => {
    const call = makeCall({ name: "shell", arguments: { command: "ls -la" } });
    expect(briefArgument(call)).toBe("ls -la");
  });

  it("uses `path` for file tools", () => {
    expect(briefArgument(makeCall({ name: "read_file", arguments: { path: "/etc/hosts" } }))).toBe("/etc/hosts");
    expect(briefArgument(makeCall({ name: "write_file", arguments: { path: "a.txt" } }))).toBe("a.txt");
    expect(briefArgument(makeCall({ name: "edit_file", arguments: { path: "b.py" } }))).toBe("b.py");
  });

  it("prefers `pattern` for glob/search_files over path-less JSON", () => {
    expect(briefArgument(makeCall({ name: "glob", arguments: { pattern: "**/*.ts" } }))).toBe("**/*.ts");
    expect(briefArgument(makeCall({ name: "search_files", arguments: { pattern: "def " } }))).toBe("def ");
  });

  it("falls back to truncated JSON for other tools", () => {
    const call = makeCall({ name: "custom_tool", argumentsRaw: longJson });
    const compact = longJson.replace(/\s+/g, " ").trim();
    expect(compact.length).toBeGreaterThan(60); // precondition for truncation
    const brief = briefArgument(call);
    expect(brief.length).toBe(61); // 60 chars + ellipsis
    expect(brief.endsWith("…")).toBe(true);
    expect(brief).toBe(`${compact.slice(0, 60)}…`);
  });

  it("does not truncate a JSON that fits in 60 characters", () => {
    const call = makeCall({ name: "custom_tool", arguments: { a: 1 } });
    expect(briefArgument(call)).toBe('{"a":1}');
  });

  it("correctly splits at the exact 60/61 character boundary", () => {
    const sixty = "x".repeat(60);
    const sixtyOne = "x".repeat(61);
    expect(briefArgument(makeCall({ name: "custom_tool", argumentsRaw: sixty }))).toBe(sixty);
    expect(briefArgument(makeCall({ name: "custom_tool", argumentsRaw: sixtyOne }))).toBe("x".repeat(60) + "…");
  });

  it("handles missing/non-string argument values and empty args", () => {
    // Numeric command is not usable as a brief -> falls through to the JSON form.
    expect(briefArgument(makeCall({ name: "shell", arguments: { command: 42 } }))).toBe('{"command":42}');
    // undefined values are dropped by JSON.stringify -> "{}".
    expect(briefArgument(makeCall({ name: "read_file", arguments: { path: undefined } }))).toBe("{}");
    expect(briefArgument(makeCall({ name: "glob", arguments: {} }))).toBe("{}");
    expect(briefArgument(makeCall({ name: "shell", arguments: {}, argumentsRaw: "" }))).toBe("{}");
  });
});

describe("prettyJson", () => {
  it("pretty-prints argumentsRaw", () => {
    const call = makeCall({ name: "shell", argumentsRaw: '{"a":1,"b":2}' });
    expect(JSON.parse(prettyJson(call))).toEqual({ a: 1, b: 2 });
  });

  it("falls back to the structured arguments when raw JSON is absent or invalid", () => {
    const call = makeCall({ name: "shell", argumentsRaw: "{broken", arguments: { x: 1 } });
    expect(JSON.parse(prettyJson(call))).toEqual({ x: 1 });
    const empty = makeCall({ name: "shell", argumentsRaw: "", arguments: {} });
    expect(prettyJson(empty)).toBe("{}");
  });
});

describe("shellOutput", () => {
  it("returns empty placeholder when there is no result", () => {
    expect(shellOutput(makeCall({ result: null }))).toEqual({
      stdout: "",
      stderr: "",
      exitCode: null,
      error: "",
    });
  });

  it("extracts stdout/stderr/exit_code from an object output", () => {
    const call = makeCall({
      result: { success: true, output: { stdout: "ok\n", stderr: "", exit_code: 0 } },
    });
    expect(shellOutput(call)).toEqual({ stdout: "ok\n", stderr: "", exitCode: 0, error: "" });
  });

  it("treats a string output as stdout", () => {
    const call = makeCall({ result: { success: true, output: "plain output" } });
    expect(shellOutput(call).stdout).toBe("plain output");
  });

  it("keeps the error message alongside the output", () => {
    const call = makeCall({
      result: { success: false, output: { stderr: "boom" }, error: "non-zero exit" },
    });
    expect(shellOutput(call).error).toBe("non-zero exit");
    expect(shellOutput(call).stderr).toBe("boom");
  });

  it("tolerates null/absent exit_code", () => {
    const call = makeCall({
      result: { success: true, output: { stdout: "x", exit_code: null } },
    });
    expect(shellOutput(call).exitCode).toBeNull();
  });
});

describe("fileEdits", () => {
  it("builds one pair from write_file args using `content` as new text", () => {
    const call = makeCall({
      name: "write_file",
      arguments: { path: "f.txt", content: "hello\nworld" },
    });
    expect(fileEdits(call)).toEqual([
      { path: "f.txt", oldText: "", newText: "hello\nworld" },
    ]);
  });

  it("builds one pair from edit_file args", () => {
    const call = makeCall({
      name: "edit_file",
      arguments: { path: "f.txt", old_text: "old\nline", new_text: "new\nline" },
    });
    expect(fileEdits(call)).toEqual([
      { path: "f.txt", oldText: "old\nline", newText: "new\nline" },
    ]);
  });

  it("parses the multi_edit edits array, honoring per-edit paths", () => {
    const call = makeCall({
      name: "multi_edit",
      arguments: {
        path: "default.txt",
        edits: [
          { path: "a.txt", old_text: "a1", new_text: "a2" },
          { old_text: "b1", new_text: "b2" },
          '{"old_text":"c1","new_text":"c2"}',
          { path: "new.txt", content: "fresh content" },
          { malformed: true },
        ],
      },
    });
    const edits = fileEdits(call);
    expect(edits).toEqual([
      { path: "a.txt", oldText: "a1", newText: "a2" },
      { path: "default.txt", oldText: "b1", newText: "b2" },
      { path: "default.txt", oldText: "c1", newText: "c2" },
      { path: "new.txt", oldText: "", newText: "fresh content" },
      { path: "default.txt", oldText: "", newText: "" },
    ]);
  });

  it("returns [] for empty args", () => {
    expect(fileEdits(makeCall({ name: "write_file", arguments: {} }))).toEqual([]);
  });
});

describe("editDiff", () => {
  it("computes an old -> new line diff inline", () => {
    const lines = editDiff({ path: "f", oldText: "a\nb", newText: "a\nc" });
    expect(lines.map((l) => [l.type, l.text])).toEqual([
      ["context", "a"],
      ["remove", "b"],
      ["add", "c"],
    ]);
  });
});

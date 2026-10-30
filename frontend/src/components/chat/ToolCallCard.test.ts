import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ToolCallCard from "./ToolCallCard.vue";
import ApprovalInline from "./ApprovalInline.vue";
import type { ToolCallView } from "./types";

function makeCall(partial: Partial<ToolCallView>): ToolCallView {
  return {
    id: "t1",
    name: "shell",
    arguments: {},
    argumentsRaw: "",
    result: null,
    approval: null,
    status: "pending",
    ...partial,
  };
}

function mountCard(call: ToolCallView, extraProps: Record<string, unknown> = {}) {
  return mount(ToolCallCard, { props: { call, ...extraProps } });
}

describe("ToolCallCard", () => {
  it.each([
    ["read_file", "read"],
    ["list_dir", "read"],
    ["glob", "search"],
    ["search_files", "search"],
    ["write_file", "edit"],
    ["edit_file", "edit"],
    ["multi_edit", "edit"],
    ["remove_file", "remove"],
    ["shell", "shell"],
    ["run_agent", "agent"],
    ["http_get", "web"],
    ["knowledge_search", "knowledge"],
    ["mcp__server__tool", "other"],
  ])("shows the %s call with the %s icon", (name, kind) => {
    const wrapper = mountCard(makeCall({ name }));
    expect(wrapper.get('[data-testid="tool-icon"]').attributes("data-kind")).toBe(kind);
  });

  it("shows the brief argument for a shell call", () => {
    const wrapper = mountCard(makeCall({ name: "shell", arguments: { command: "npm test" } }));
    expect(wrapper.get('[data-testid="tool-brief"]').text()).toContain("npm test");
  });

  it("shows the path for read_file calls", () => {
    const wrapper = mountCard(makeCall({ name: "read_file", arguments: { path: "src/main.py" } }));
    expect(wrapper.get('[data-testid="tool-brief"]').text()).toContain("src/main.py");
  });

  it("shows the pattern for glob calls", () => {
    const wrapper = mountCard(makeCall({ name: "glob", arguments: { pattern: "**/*.ts" } }));
    expect(wrapper.get('[data-testid="tool-brief"]').text()).toContain("**/*.ts");
  });

  it("renders the awaiting_approval status badge", () => {
    const wrapper = mountCard(makeCall({ status: "awaiting_approval" }));
    expect(wrapper.get('[data-testid="tool-status"]').text()).toBe("Требуется подтверждение");
  });

  it("renders the error status badge", () => {
    const wrapper = mountCard(makeCall({ status: "error" }));
    expect(wrapper.get('[data-testid="tool-status"]').text()).toBe("Ошибка");
  });

  it.each([
    [true, "completed", "Выполнено"],
    [false, "error", "Ошибка"],
  ] as const)("keeps the collapsed result in the header (%s)", (success, status, label) => {
    const wrapper = mountCard(makeCall({
      status,
      result: { success, output: "result", error: success ? undefined : "failed" },
    }));
    expect(wrapper.get('[data-testid="tool-status"]').text()).toBe(label);
    expect(wrapper.find(".tool-result-hint").exists()).toBe(false);
    expect(wrapper.find('[data-testid="tool-detail"]').exists()).toBe(false);
  });

  it("renders ApprovalInline when the call awaits approval", () => {
    const wrapper = mountCard(
      makeCall({
        status: "awaiting_approval",
        approval: {
          id: "ap1",
          tool_call_id: "t1",
          tool_name: "shell",
          arguments: { command: "rm -rf" },
          risk_level: "high",
          status: "awaiting",
          comment: null,
        },
      }),
    );
    expect(wrapper.find('[data-testid="approval"]').exists()).toBe(true);
  });

  it("forwards approve and reject events from ApprovalInline", async () => {
    const wrapper = mountCard(
      makeCall({
        status: "awaiting_approval",
        approval: {
          id: "ap1",
          tool_call_id: "t1",
          tool_name: "shell",
          arguments: {},
          risk_level: "low",
          status: "awaiting",
          comment: null,
        },
      }),
    );
    await wrapper.get('[data-testid="approve-btn"]').trigger("click");
    expect(wrapper.emitted("approve")?.[0]).toEqual(["ap1"]);

    await wrapper.get('[data-testid="reject-btn"]').trigger("click");
    await wrapper.get('[data-testid="reject-comment"]').setValue("нет");
    await wrapper.get('[data-testid="reject-submit"]').trigger("click");
    expect(wrapper.emitted("reject")?.[0]).toEqual([{ approvalId: "ap1", comment: "нет" }]);
  });

  it("does not render ApprovalInline for a completed call", () => {
    const wrapper = mountCard(makeCall({ status: "completed" }));
    expect(wrapper.find('[data-testid="approval"]').exists()).toBe(false);
  });

  it("shows shell stdout/stderr/exit code in the brief tab", async () => {
    const wrapper = mountCard(
      makeCall({
        name: "shell",
        arguments: { command: "make" },
        result: { success: true, output: { stdout: "built\n", stderr: "warn", exit_code: 2 } },
        status: "completed",
      }),
    );
    await wrapper.get('[data-testid="tool-call"] .tool-head').trigger("click");
    expect(wrapper.get('[data-testid="shell-stdout"]').text()).toContain("built");
    expect(wrapper.get('[data-testid="shell-stderr"]').text()).toContain("warn");
    expect(wrapper.get('[data-testid="shell-exit"]').text()).toContain("2");
    expect(wrapper.get('[data-testid="shell-command"]').text()).toBe("$ make");
  });

  it("shows the old -> new line diff for edit_file", async () => {
    const wrapper = mountCard(
      makeCall({
        name: "edit_file",
        arguments: { path: "app.py", old_text: "line A\nline B", new_text: "line A\nline C" },
      }),
    );
    await wrapper.get('[data-testid="tool-call"] .tool-head').trigger("click");
    const panel = wrapper.get('[data-testid="file-edits"]').text();
    expect(panel).toContain("app.py");
    const diff = wrapper.get('[data-testid="file-edits"] .diff').text();
    expect(diff).toContain("line B");
    expect(diff).toContain("line C");
  });

  it("shows the write_file content as new lines in the brief tab", async () => {
    const wrapper = mountCard(
      makeCall({
        name: "write_file",
        arguments: { path: "out.txt", content: "первая\nвторая строка\nтретья" },
      }),
    );
    await wrapper.get('[data-testid="tool-call"] .tool-head').trigger("click");
    const diff = wrapper.get('[data-testid="file-edits"] .diff').text();
    // No old text — every content line must be present as a "+" addition so the
    // whole payload is visible as the file's new text.
    expect(diff).toContain("первая");
    expect(diff).toContain("вторая строка");
    expect(diff).toContain("третья");
    // Must be additions, not removals of a phantom old side.
    expect(diff.replace(/^\+/gm, "")).not.toHaveLength(0);
  });

  it("forwards approval hotkeys to the inner ApprovalInline", async () => {
    const wrapper = mountCard(
      makeCall({
        status: "awaiting_approval",
        approval: {
          id: "ap1",
          tool_call_id: "t1",
          tool_name: "shell",
          arguments: {},
          risk_level: "low",
          status: "awaiting",
          comment: null,
        },
      }),
      { hotkeys: true },
    );
    const inline = wrapper.findComponent(ApprovalInline);
    expect(inline.exists()).toBe(true);
    expect(inline.props("hotkeys")).toBe(true);
    // With hotkeys armed, the A key on the window resolves the approval
    // through the forwarded approve event.
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "a", bubbles: true, cancelable: true }));
    expect(wrapper.emitted("approve")?.[0]).toEqual(["ap1"]);
  });

  it("defaults hotkeys to false when not provided", () => {
    const wrapper = mountCard(
      makeCall({
        status: "awaiting_approval",
        approval: {
          id: "ap1",
          tool_call_id: "t1",
          tool_name: "shell",
          arguments: {},
          risk_level: "low",
          status: "awaiting",
          comment: null,
        },
      }),
    );
    expect(wrapper.findComponent(ApprovalInline).props("hotkeys")).toBe(false);
  });

  it("switches to a pretty JSON tab", async () => {
    const wrapper = mountCard(
      makeCall({ name: "shell", arguments: { command: "ls" }, argumentsRaw: '{"command":"ls"}' }),
    );
    await wrapper.get('[data-testid="tool-call"] .tool-head').trigger("click");
    await wrapper.get('[data-testid="tab-json"]').trigger("click");
    expect(wrapper.get('[data-testid="json-panel"]').text()).toContain('"command"');
  });

  it("collapses/expands on the header click", async () => {
    const wrapper = mountCard(makeCall({ name: "shell", arguments: { command: "x" } }));
    expect(wrapper.find('[data-testid="tool-detail"]').exists()).toBe(false);
    await wrapper.get('[data-testid="tool-call"] .tool-head').trigger("click");
    expect(wrapper.find('[data-testid="tool-detail"]').exists()).toBe(true);
    await wrapper.get('[data-testid="tool-call"] .tool-head').trigger("click");
    expect(wrapper.find('[data-testid="tool-detail"]').exists()).toBe(false);
  });
});

it("renders a neutral patch with its actual application status", async () => {
  const wrapper = mountCard(makeCall({
    name: "engine.diff", status: "completed",
    result: { success: true, patch: "+new\n-old", paths: ["a.txt"], status: "proposed" },
  }));
  await wrapper.get('.tool-head').trigger("click");
  expect(wrapper.get('[data-testid="engine-patch"]').text()).toContain("+new");
  expect(wrapper.text()).toContain("Предложено");
  expect(wrapper.text()).toContain("a.txt");
});

it("labels a turn-scoped approval and forwards only that request", async () => {
  const wrapper = mountCard(makeCall({ status: "awaiting_approval", approval: {
    id: "native", tool_call_id: "call", tool_name: "write", arguments: {}, risk_level: "observed",
    status: "pending", comment: null, executor: "engine", generation: 3, scope: "turn", request_id: "req",
  } }));
  expect(wrapper.text()).toContain("до конца хода");
  await wrapper.get('[data-testid="approve-btn"]').trigger("click");
  expect(wrapper.emitted("approve")).toEqual([["native"]]);
  await wrapper.get('[data-testid="reject-btn"]').trigger("click");
  await wrapper.get('[data-testid="reject-submit"]').trigger("click");
  expect(wrapper.emitted("reject")).toEqual([[{ approvalId: "native", comment: "" }]]);
});

describe("ToolCallCard results", () => {
  it("shows a read file in its language with line numbers", async () => {
    const wrapper = mountCard(
      makeCall({
        name: "read_file",
        arguments: { path: "src/a.py" },
        status: "completed",
        result: { success: true, output: { path: "src/a.py", start_line: 5, content: "x = 1\n" } },
      }),
    );
    await wrapper.get(".tool-head").trigger("click");
    const view = wrapper.get('[data-testid="result-blocks"] [data-testid="code-view"]');
    expect(view.attributes("data-lang")).toBe("python");
    expect(view.text()).toContain("x = 1");
    expect(view.get('[data-testid="code-number"]').text()).toBe("5");
  });

  it("shows any other tool output as a code block, not only the brief line", async () => {
    const wrapper = mountCard(
      makeCall({ name: "mcp__srv__tool", status: "completed", result: { success: true, output: { rows: [1] } } }),
    );
    await wrapper.get(".tool-head").trigger("click");
    expect(wrapper.get('[data-testid="result-blocks"] [data-testid="code-view"]').attributes("data-lang")).toBe("json");
  });
});

import { describe, expect, it } from "vitest";
import {
  toChatTimelineItems,
  toChatApproval,
  toChatToolResult,
  toChatMessage,
  toChatToolCallView,
  toChildSessions,
  toLayoutContext,
  toLayoutSessionTree,
  toModelOptions,
  toProjectOptions,
  toToolEntries,
} from "./uiAdapters";
import type {
  Message,
  Session,
  SessionNode,
  ToolCallView,
  Approval,
  OutboundAudit,
  ContextStats as ApiContextStats,
} from "./api/types";

function apiMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: 1,
    session_id: "s1",
    role: "user",
    content: "привет",
    tool_call_id: null,
    tool_calls: null,
    token_count: null,
    skipped: false,
    kind: "normal",
    reasoning_content: null,
    reasoning_effort: null,
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function apiSession(id: string, overrides: Partial<Session> = {}): Session {
  return {
    id,
    project_id: "p1",
    parent_id: null,
    mode: "dev",
    llm_profile: "default",
    title: id,
    status: "completed",
    error: null,
    archived: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function apiApproval(overrides: Partial<Approval> = {}): Approval {
  return {
    id: "ap1",
    tool_call_id: "call_1",
    tool_name: "shell",
    arguments: {},
    risk_level: "low",
    status: "pending",
    comment: null,
    ...overrides,
  };
}

describe("toChatMessage", () => {
  it("maps api message fields and coerces null content to empty string", () => {
    const msg = apiMessage({ content: null, role: "assistant", kind: "assistant" });
    const chat = toChatMessage(msg);
    expect(chat.id).toBe("1");
    expect(chat.role).toBe("assistant");
    expect(chat.kind).toBe("assistant");
    expect(chat.content).toBe("");
    expect(chat.reasoning_content).toBeNull();
  });

  it("maps id number to string", () => {
    expect(toChatMessage(apiMessage({ id: 42 })).id).toBe("42");
  });
});

describe("toChatToolCallView", () => {
  it("maps result, approval and status", () => {
    const view: ToolCallView = {
      id: "c1",
      name: "shell",
      arguments: { command: "ls" },
      argumentsRaw: '{"command":"ls"}',
      result: { success: true, output: "ok" },
      approval: apiApproval(),
      status: "awaiting_approval",
    };
    const chat = toChatToolCallView(view);
    expect(chat.id).toBe("c1");
    expect(chat.result?.output).toBe("ok");
    expect(chat.approval?.id).toBe("ap1");
    expect(chat.status).toBe("awaiting_approval");
  });
});

describe("toChatTimelineItems", () => {
  it("maps all kinds including summary and streaming", () => {
    const items = [
      { kind: "user" as const, message: apiMessage() },
      { kind: "assistant" as const, message: apiMessage({ role: "assistant", kind: "assistant" }) },
      {
        kind: "tool" as const,
        call: {
          id: "c1",
          name: "shell",
          arguments: {},
          argumentsRaw: "{}",
          result: null,
          approval: null,
          status: "pending" as const,
        },
      },
      { kind: "summary" as const, message: apiMessage({ kind: "summary" }) },
      { kind: "streaming" as const, content: "текст", reasoning: "думаю" },
    ];
    const chat = toChatTimelineItems(items);
    expect(chat.map((i) => i.kind)).toEqual(["user", "assistant", "tool", "summary", "streaming"]);
    expect(chat[0].kind).toBe("user");
    if (chat[0].kind === "user") expect(chat[0].message.id).toBe("1");
  });
});

describe("layout adapters", () => {
  it("toModelOptions filters unconfigured profiles", () => {
    const config = {
      llm_configured: true,
      llm_model: "m",
      llm_destination: "http://local",
      knowledge_base_enabled: false,
      network_tools_enabled: true,
      network_allowlist: [],
      api_auth_enabled: true,
      execution_mode: "embedded" as const,
      default_llm_profile: "p1",
      llm_profiles: [
        { name: "p1", model: "model-a", destination: "dst", configured: true },
        { name: "p2", model: "model-b", destination: "dst", configured: false },
      ],
    };
    expect(toModelOptions(config)).toEqual([{ id: "p1", label: "p1 · model-a" }]);
    expect(toModelOptions(null)).toEqual([]);
  });

  it("toLayoutSessionTree maps nested sessions recursively", () => {
    const node: SessionNode = {
      session: apiSession("root"),
      children: [
        { session: apiSession("child", { title: "ребёнок" }), children: [] },
      ],
    };
    const tree = toLayoutSessionTree([node]);
    expect(tree[0].id).toBe("root");
    expect(tree[0].children[0].id).toBe("child");
    expect(tree[0].children[0].title).toBe("ребёнок");
  });

  it("toLayoutContext maps nullable and passes through fields", () => {
    expect(toLayoutContext(null)).toBeNull();
    const ctx = toLayoutContext({
      approximate_tokens: 10,
      omitted_messages: 2,
      truncated_tool_results: 0,
      context_window: 100,
      reserved_tokens: 20,
    });
    expect(ctx?.approximate_tokens).toBe(10);
    expect(ctx?.omitted_messages).toBe(2);
    expect(ctx?.context_window).toBe(100);
  });

  it("toLayoutContext passes through specific new fields", () => {
    const apiStats: ApiContextStats = {
      approximate_tokens: 10,
      omitted_messages: 2,
      truncated_tool_results: 0,
      reserved_tokens: 20,
      context_window: 100,
      input_budget: 500,
      fits: false,
      error: "some error",
    };
    const ctx = toLayoutContext(apiStats);
    expect(ctx?.input_budget).toBe(500);
    expect(ctx?.fits).toBe(false);
    expect(ctx?.error).toBe("some error");
  });

  it("toLayoutContext handles null/undefined for optional targets correctly", () => {
    const apiStats: ApiContextStats = {
      approximate_tokens: 10,
      omitted_messages: 2,
      truncated_tool_results: 0,
      reserved_tokens: 20,
      context_window: 100,
      input_budget: null,
      fits: null,
    };
    const ctx = toLayoutContext(apiStats);
    expect(ctx?.input_budget).toBeUndefined();
    expect(ctx?.fits).toBeUndefined();
  });

  it("toChildSessions maps id/title/status", () => {
    const children = toChildSessions([apiSession("c1", { status: "running" })]);
    expect(children).toEqual([{ id: "c1", title: "c1", status: "running" }]);
  });

  it("toToolEntries lists only available tools with enabled state from config", () => {
    const tools = toToolEntries({
      llm_configured: true,
      llm_model: "m",
      llm_destination: "dst",
      knowledge_base_enabled: false,
      network_tools_enabled: true,
      network_allowlist: [],
      api_auth_enabled: true,
      execution_mode: "embedded",
      default_llm_profile: "p",
      llm_profiles: [],
    });
    expect(tools.some((t) => t.name === "shell" && t.enabled)).toBe(true);
    expect(tools.find((t) => t.name === "http_get")?.enabled).toBe(true);
    expect(tools.find((t) => t.name === "knowledge_search")?.enabled).toBe(false);
  });
});

it("keeps native patch and approval scope across the timeline adapter", () => {
  const request = { id: "approval", tool_call_id: "call", tool_name: "write", arguments: {}, risk_level: "observed", status: "pending", comment: null, executor: "engine" as const, generation: 2, scope: "turn" as const, request_id: "native" };
  const result = { success: true, patch: "+new", paths: ["a.txt"], status: "proposed" as const };
  expect(toChatApproval(request)).toEqual(request);
  expect(toChatToolResult(result)).toEqual(result);
});

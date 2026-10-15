import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import { api } from "../api/client";
import type {
  Approval,
  ContextStats,
  Message,
  ServerEvent,
  Session,
  TimelineItem,
  ToolCallRaw,
} from "../api/types";
import { useSessionStore } from "./session";

function msg(id: number, overrides: Partial<Message> = {}): Message {
  return {
    id,
    session_id: "s1",
    role: "user",
    content: null,
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

function toolCall(id: string, name: string, args: string): ToolCallRaw {
  return { id, type: "function", function: { name, arguments: args } };
}

function approval(overrides: Partial<Approval> = {}): Approval {
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

function delta(
  type: "message.delta" | "reasoning.delta" | "stream.reset",
  streamId: string,
  seq: number | undefined,
  deltaText: string,
): ServerEvent {
  return {
    type,
    session_id: "s1",
    ephemeral: true,
    payload: { stream_id: streamId, ...(seq === undefined ? {} : { seq }), ...(deltaText ? { delta: deltaText } : {}) },
  };
}

function persistent(type: string, payload: Record<string, unknown>): ServerEvent {
  return { id: 5, type, session_id: "s1", payload, created_at: "2026-01-01T00:00:00Z" };
}

function sessionStatus(payload: Record<string, unknown>): ServerEvent {
  return { id: 6, type: "session.status", session_id: "s1", payload, created_at: "2026-01-01T00:00:00Z" };
}

function countKinds(items: TimelineItem[]): string[] {
  return items.map((item) => item.kind);
}

describe("session store timeline", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  it("maps user, assistant, tool call with result, approval and summary", () => {
    const store = useSessionStore();
    store.messages = [
      msg(1, { role: "user", content: "Привет" }),
      msg(2, {
        role: "assistant",
        content: "Сейчас посмотрю",
        tool_calls: [toolCall("call_1", "read_file", `{"path":"a.txt"}`)],
        reasoning_content: "думаю…",
      }),
      msg(3, {
        role: "tool",
        tool_call_id: "call_1",
        content: JSON.stringify({ success: true, output: "data" }),
      }),
      msg(4, { role: "user", kind: "summary", content: "Контекст…" }),
    ];
    store.approvals = [approval({ status: "approved" })];

    const items = store.timeline;
    expect(countKinds(items)).toEqual(["user", "assistant", "tool", "summary"]);

    const userItem = items[0];
    if (userItem.kind === "user") expect(userItem.message.content).toBe("Привет");

    const toolItem = items[2];
    if (toolItem.kind === "tool") {
      expect(toolItem.call.id).toBe("call_1");
      expect(toolItem.call.name).toBe("read_file");
      expect(toolItem.call.arguments).toEqual({ path: "a.txt" });
      expect(toolItem.call.argumentsRaw).toBe(`{"path":"a.txt"}`);
      expect(toolItem.call.result).toEqual({ success: true, output: "data" });
      expect(toolItem.call.approval?.status).toBe("approved");
      expect(toolItem.call.status).toBe("completed");
    }
  });

  it("marks a pending approval as awaiting_approval and parses errors", () => {
    const store = useSessionStore();
    store.messages = [
      msg(2, {
        role: "assistant",
        content: null,
        tool_calls: [toolCall("call_1", "shell", `{"command":"rm -rf /"}`)],
      }),
    ];
    store.approvals = [approval()];

    const items = store.timeline;
    // Content is empty so only the tool card is emitted, not a separate assistant bubble.
    expect(countKinds(items)).toEqual(["tool"]);
    const item = items[0];
    if (item.kind === "tool") {
      expect(item.call.status).toBe("awaiting_approval");
      expect(item.call.approval?.risk_level).toBe("low");
    }
  });

  it("derives tools without a result as pending and failures as error/rejected", () => {
    const store = useSessionStore();
    store.messages = [
      msg(2, {
        role: "assistant",
        content: "",
        tool_calls: [
          toolCall("a", "glob", `{"pattern":"**/*.ts"}`),
          toolCall("b", "shell", `{"command":"no"}`),
        ],
      }),
      msg(3, {
        role: "tool",
        tool_call_id: "b",
        content: JSON.stringify({ success: false, error: "нет файла" }),
      }),
    ];

    const items = store.timeline;
    const toolItems = items.filter((i) => i.kind === "tool");
    expect(toolItems).toHaveLength(2);
    const [pendingItem, errorItem] = toolItems;
    if (pendingItem.kind === "tool" && errorItem.kind === "tool") {
      expect(pendingItem.call.status).toBe("pending");
      expect(pendingItem.call.result).toBeNull();
      expect(errorItem.call.status).toBe("error");
      expect(errorItem.call.result).toEqual({ success: false, error: "нет файла" });
    }
  });

  it("hides skipped messages but keeps their tool results for the shown call", () => {
    const store = useSessionStore();
    store.messages = [
      msg(1, { role: "user", content: "visible", skipped: true }),
      msg(2, {
        role: "assistant",
        content: "готово",
        tool_calls: [toolCall("c1", "shell", "{}")],
      }),
      msg(3, { role: "tool", tool_call_id: "c1", content: JSON.stringify({ success: true }) }),
    ];

    expect(countKinds(store.timeline)).toEqual(["assistant", "tool"]);
    const toolItem = store.timeline[1];
    if (toolItem.kind === "tool") {
      expect(toolItem.call.result).toEqual({ success: true });
    }
  });

  it("appends a streaming item while a stream is active", () => {
    const store = useSessionStore();
    store.streaming = { content: "текс", reasoning: "" };
    expect(countKinds(store.timeline)).toEqual(["streaming"]);
    const item = store.timeline[0];
    if (item.kind === "streaming") {
      expect(item.content).toBe("текс");
      expect(item.reasoning).toBe("");
    }
  });

  it("does not emit an empty streaming item", () => {
    const store = useSessionStore();
    store.streaming = { content: "", reasoning: "" };
    expect(store.timeline).toEqual([]);
  });

  it("keeps the order stable for an out-of-order message list", () => {
    const store = useSessionStore();
    store.messages = [
      msg(5, { role: "assistant", content: "второе" }),
      msg(2, { role: "user", content: "первое" }),
      msg(8, { role: "user", content: "третье" }),
    ];
    // Timeline follows the array order, not the ids.
    expect(store.timeline.map((i) => (i.kind === "user" || i.kind === "assistant" ? i.message.content : null))).toEqual([
      "второе",
      "первое",
      "третье",
    ]);
  });

  it("does not crash on malformed tool results or arguments from the database", () => {
    const store = useSessionStore();
    // Malformed rows bypass the TS types via `unknown`, as a corrupted DB row would.
    const malformedCalls = [
      { id: "a", type: "function", function: { name: "shell", arguments: "not json{{" } },
      { id: "b", type: "function", function: { name: null, arguments: 42 } },
      { id: "c" },
    ] as unknown as ToolCallRaw[];
    store.messages = [
      msg(2, {
        role: "assistant",
        content: null,
        tool_calls: malformedCalls,
      }),
      msg(3, { role: "tool", tool_call_id: "a", content: "not json" }),
      msg(4, { role: "tool", tool_call_id: "b", content: JSON.stringify({ output: "ok" }) }), // missing success
    ];

    const items = store.timeline;
    expect(items).toHaveLength(3); // one tool card per call, nothing crashes
    const [a, b, c] = items;
    if (a.kind === "tool") {
      expect(a.call.id).toBe("a");
      expect(a.call.argumentsRaw).toBe("not json{{");
      expect(a.call.arguments).toEqual({ _invalid_arguments: "not json{{" });
      expect(a.call.result).toBeNull();
      expect(a.call.status).toBe("pending");
    }
    if (b.kind === "tool") {
      // Non-string arguments are treated as empty; a result without `success` is ignored.
      expect(b.call.argumentsRaw).toBe("");
      expect(b.call.result).toBeNull();
      expect(b.call.status).toBe("pending");
    }
    if (c.kind === "tool") {
      expect(c.call.id).toBe("c");
      expect(c.call.name).toBe("tool"); // fallback name
    }
  });

  it("keeps the last seen result for a tool call across duplicate tool messages", () => {
    const store = useSessionStore();
    store.messages = [
      msg(2, { role: "assistant", tool_calls: [toolCall("x", "shell", "{}")], content: null }),
      msg(3, { role: "tool", tool_call_id: "x", content: JSON.stringify({ success: true, output: "1" }) }),
      msg(4, { role: "tool", tool_call_id: "x", content: JSON.stringify({ success: true, output: "2" }) }),
    ];
    const toolItem = store.timeline[0];
    if (toolItem.kind === "tool") {
      expect(toolItem.call.result).toEqual({ success: true, output: "2" });
    }
  });
});

describe("session store applyEvent", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  it("appends message deltas and drops repeats or stale seq", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent(delta("message.delta", "st1", 1, "A"));
    store.applyEvent(delta("message.delta", "st1", 1, "dup"));
    store.applyEvent(delta("message.delta", "st1", 2, "B"));
    store.applyEvent(delta("message.delta", "st1", 0, "old"));
    expect(store.streaming?.content).toBe("AB");

    store.applyEvent(delta("reasoning.delta", "st1", 3, "r1"));
    store.applyEvent(delta("reasoning.delta", "st1", 3, "dup"));
    store.applyEvent(delta("reasoning.delta", "st1", 4, "r2"));
    expect(store.streaming?.reasoning).toBe("r1r2");
  });

  it("starts a new stream on a different stream_id and drops stale seq from the old one", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent(delta("message.delta", "old", 1, "start"));
    store.applyEvent(delta("message.delta", "new", 1, "fresh"));
    // A late delta from the previous stream must not touch the new one.
    store.applyEvent(delta("message.delta", "old", 2, "late"));
    expect(store.streaming?.content).toBe("fresh");
  });

  it("clears the stream on stream.reset", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent(delta("message.delta", "st1", 1, "text"));
    store.applyEvent(delta("stream.reset", "st1", undefined, ""));
    expect(store.streaming).toBeNull();
  });

  it("appends to the current stream when seq is omitted", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent(delta("message.delta", "st1", undefined, "x"));
    store.applyEvent(delta("message.delta", "st1", undefined, "y"));
    expect(store.streaming?.content).toBe("xy");
  });

  it("does not wipe the live stream when a new stream_id arrives without a string delta", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent(delta("message.delta", "st1", 1, "keep"));

    // A broken frame: new stream_id but no string delta must be ignored, not
    // treated as the start of a fresh generation (self-check point 1).
    store.applyEvent({
      type: "message.delta",
      session_id: "s1",
      ephemeral: true,
      payload: { stream_id: "other", seq: 1, delta: null },
    } as unknown as ServerEvent);
    store.applyEvent({
      type: "message.delta",
      session_id: "s1",
      ephemeral: true,
      payload: { stream_id: "st1", seq: 2, delta: "" },
    } as unknown as ServerEvent);

    expect(store.streaming?.content).toBe("keep"); // live stream untouched
  });

  it("ignores deltas for a session that is not open", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent({
      type: "message.delta",
      session_id: "other",
      ephemeral: true,
      payload: { stream_id: "st", seq: 1, delta: "x" },
    });
    expect(store.streaming).toBeNull();
  });

  it("message.created clears streaming and reloads messages", async () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.streaming = { content: "x", reasoning: "" };
    const spy = vi
      .spyOn(api, "messages")
      .mockResolvedValue([msg(9, { role: "assistant", content: "готово" })]);

    store.applyEvent(persistent("message.created", { role: "assistant", message_id: 9 }));
    await flushPromises();

    expect(store.streaming).toBeNull();
    expect(spy).toHaveBeenCalledWith("s1");
    expect(store.messages.map((m) => m.content)).toEqual(["готово"]);
  });

  it("reloads approvals on approval.required and tool.completed reloads messages", async () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    const approvalsSpy = vi.spyOn(api, "approvals").mockResolvedValue([approval()]);
    const messagesSpy = vi.spyOn(api, "messages").mockResolvedValue([]);

    store.applyEvent(persistent("approval.required", { status: "awaiting_confirmation" }));
    await flushPromises();
    expect(approvalsSpy).toHaveBeenCalledWith("s1");
    expect(store.approvals).toHaveLength(1);

    store.applyEvent(persistent("tool.completed", { tool_call_id: "a", name: "shell", status: "completed" }));
    await flushPromises();
    expect(messagesSpy).toHaveBeenCalledWith("s1");
  });

  it("applies session.status and session.error to the open session", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.session = {
      id: "s1",
      project_id: "p1",
      parent_id: null,
      mode: "dev",
      llm_profile: "default",
      title: "t",
      status: "pending",
      error: null,
      archived: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };

    store.applyEvent(sessionStatus({ status: "running" }));
    expect(store.session?.status).toBe("running");

    store.applyEvent({ id: 7, type: "session.error", session_id: "s1", payload: { status: "error", error: "boom" }, created_at: "2026-01-01T00:00:00Z" });
    expect(store.session?.status).toBe("error");
    expect(store.session?.error).toBe("boom");
  });

  it("does not throw on a message.delta with a missing or null payload and keeps the live stream", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.applyEvent(delta("message.delta", "st1", 1, "keep"));

    // A frame without a payload (malformed server data) must not throw or truncate.
    expect(() =>
      store.applyEvent({
        type: "message.delta",
        session_id: "s1",
        ephemeral: true,
      } as unknown as ServerEvent),
    ).not.toThrow();
    expect(() =>
      store.applyEvent({
        type: "message.delta",
        session_id: "s1",
        ephemeral: true,
        payload: null,
      } as unknown as ServerEvent),
    ).not.toThrow();
    expect(() =>
      store.applyEvent({
        type: "reasoning.delta",
        session_id: "s1",
        ephemeral: true,
        payload: { delta: "x" }, // stream_id missing
      } as unknown as ServerEvent),
    ).not.toThrow();
    expect(store.streaming?.content).toBe("keep"); // unaffected by bad frames
  });

  it("does not throw on session.status with a null payload", () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    store.session = {
      id: "s1",
      project_id: "p1",
      parent_id: null,
      mode: "dev",
      llm_profile: "default",
      title: "t",
      status: "pending",
      error: null,
      archived: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };

    expect(() =>
      store.applyEvent({ id: 9, type: "session.status", session_id: "s1", payload: null, created_at: "2026-01-01T00:00:00Z" } as unknown as ServerEvent),
    ).not.toThrow();
    expect(store.session?.status).toBe("pending"); // unchanged
  });

  it("ignores events for a different session", async () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    const messagesSpy = vi.spyOn(api, "messages").mockResolvedValue([]);

    store.applyEvent({ id: 8, type: "message.created", session_id: "other", payload: {}, created_at: "2026-01-01T00:00:00Z" });
    await flushPromises();
    expect(messagesSpy).not.toHaveBeenCalled();
  });

  it("reloads context on context.compacted", async () => {
    const store = useSessionStore();
    store.sessionId = "s1";
    const spy = vi.spyOn(api, "context").mockResolvedValue({
      approximate_tokens: 100,
      omitted_messages: 2,
      truncated_tool_results: 0,
      context_window: 1000,
      reserved_tokens: 100,
    });

    store.applyEvent(persistent("context.compacted", { omitted_messages: 2 }));
    await flushPromises();
    expect(spy).toHaveBeenCalledWith("s1");
    expect(store.context?.omitted_messages).toBe(2);
  });
});

describe("session store actions", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  it("send() forwards reasoningEffort, resets streaming and adds the message exactly once across a reload race", async () => {
    const sendSpy = vi
      .spyOn(api, "sendMessage")
      .mockResolvedValue(msg(42, { role: "user", content: "задача" }));
    const messagesSpy = vi
      .spyOn(api, "messages")
      .mockResolvedValue([msg(42, { role: "user", content: "задача" })]); // reload already contains it
    const store = useSessionStore();
    store.sessionId = "s1";
    store.streaming = { content: "x", reasoning: "y" };

    // Simulate message.created racing: reload the list BEFORE send() appends.
    store.applyEvent(persistent("message.created", { role: "user", message_id: 42 }));
    await flushPromises();
    expect(messagesSpy).toHaveBeenCalledTimes(1);

    await store.send("задача", "high");
    expect(sendSpy).toHaveBeenCalledWith("s1", "задача", "high");

    // Dedup by id: the reloaded copy and the local append must not duplicate.
    expect(store.messages.filter((m) => m.id === 42)).toHaveLength(1);
    expect(store.streaming).toBeNull(); // stream reset after send
  });

  it("open() loads every slice for the session", async () => {
    const sessionRecord: Session = {
      id: "s1",
      project_id: "p1",
      parent_id: null,
      mode: "dev",
      llm_profile: "default",
      title: "t",
      status: "completed",
      error: null,
      archived: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    vi.spyOn(api, "messages").mockResolvedValue([msg(1)]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    vi.spyOn(api, "context").mockResolvedValue({
      approximate_tokens: 0,
      omitted_messages: 0,
      truncated_tool_results: 0,
      context_window: 100,
      reserved_tokens: 10,
    });
    vi.spyOn(api, "session").mockResolvedValue(sessionRecord);

    const store = useSessionStore();
    await store.open("s1");

    expect(store.sessionId).toBe("s1");
    expect(store.session?.status).toBe("completed");
    expect(store.messages).toHaveLength(1);
  });
});

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function sessionRecord(id: string, overrides: Partial<Session> = {}): Session {
  return {
    id,
    project_id: "p1",
    parent_id: null,
    mode: "dev",
    llm_profile: "default",
    title: `t-${id}`,
    status: "completed",
    error: null,
    archived: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function contextStats(omitted = 0): ContextStats {
  return {
    approximate_tokens: 100,
    omitted_messages: omitted,
    truncated_tool_results: 0,
    context_window: 1000,
    reserved_tokens: 100,
  };
}

describe("session store open races", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  /** Stubs every API slice with immediate values except `messages`, which
   *  records its deferred per session id so the test can resolve them in
   *  arbitrary order. */
  function mockReloadSlicesWithDeferredMessages(): {
    messagesCalls: Array<{ id: string; d: ReturnType<typeof deferred<Message[]>> }>;
  } {
    const messagesCalls: Array<{ id: string; d: ReturnType<typeof deferred<Message[]>> }> = [];
    vi.spyOn(api, "messages").mockImplementation((id: string) => {
      const d = deferred<Message[]>();
      messagesCalls.push({ id, d });
      return d.promise;
    });
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    vi.spyOn(api, "context").mockImplementation((id: string) =>
      Promise.resolve(contextStats(id === "s2" ? 2 : 1)),
    );
    vi.spyOn(api, "session").mockImplementation((id: string) =>
      Promise.resolve(sessionRecord(id)),
    );
    return { messagesCalls };
  }

  it("open A then open B: B's reload commits first, late A reload is dropped", async () => {
    const store = useSessionStore();
    const { messagesCalls } = mockReloadSlicesWithDeferredMessages();

    const openA = store.open("A");
    const openB = store.open("B");

    const bCall = messagesCalls.find((c) => c.id === "B");
    expect(bCall).toBeDefined();
    bCall!.d.resolve([msg(2, { session_id: "B", role: "user", content: "B-msg" })]);
    await flushPromises();

    expect(store.sessionId).toBe("B");
    expect(store.messages.map((m) => m.id)).toEqual([2]);

    const aCall = messagesCalls.find((c) => c.id === "A");
    expect(aCall).toBeDefined();
    aCall!.d.resolve([msg(1, { session_id: "A", role: "user", content: "A-msg" })]);
    await flushPromises();

    expect(store.messages.map((m) => m.id)).toEqual([2]);

    await openA;
    await openB;
  });

  it("A→B→A: stale first A never wins over the freshly reopened A", async () => {
    const store = useSessionStore();
    const { messagesCalls } = mockReloadSlicesWithDeferredMessages();

    const openA1 = store.open("A");
    const openB = store.open("B");
    const openA2 = store.open("A");

    const newACall = messagesCalls.filter((c) => c.id === "A")[1];
    newACall.d.resolve([msg(30, { session_id: "A", content: "fresh-A" })]);
    await flushPromises();

    expect(store.sessionId).toBe("A");
    expect(store.messages.map((m) => m.id)).toEqual([30]);

    const bCall = messagesCalls.find((c) => c.id === "B");
    bCall!.d.resolve([msg(20, { session_id: "B", content: "B-msg" })]);
    const oldACall = messagesCalls.find((c) => c.id === "A");
    oldACall!.d.resolve([msg(10, { session_id: "A", content: "old-A" })]);
    await flushPromises();

    expect(store.messages.map((m) => m.id)).toEqual([30]);

    await openA1;
    await openB;
    await openA2;
  });

  it("a failing open(B) clears A's data and a retry genuinely re-reads B", async () => {
    const store = useSessionStore();

    vi.spyOn(api, "messages").mockResolvedValue([msg(1, { session_id: "A", content: "A" })]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    vi.spyOn(api, "context").mockResolvedValue(contextStats(1));
    vi.spyOn(api, "session").mockResolvedValue(sessionRecord("A"));
    await store.open("A");
    expect(store.sessionId).toBe("A");
    expect(store.messages).toHaveLength(1);
    expect(store.session).not.toBeNull();

    const messagesCalls: Array<ReturnType<typeof deferred<Message[]>>> = [];
    vi.spyOn(api, "messages").mockImplementation(() => {
      const d = deferred<Message[]>();
      messagesCalls.push(d);
      return d.promise;
    });
    const openB1 = store.open("B");
    expect(store.sessionId).toBe("B");
    expect(store.messages).toHaveLength(0);
    expect(store.session).toBeNull();
    expect(store.context).toBeNull();

    messagesCalls[0].reject(new Error("HTTP 404"));
    await expect(openB1).rejects.toThrow("HTTP 404");

    expect(store.messages).toHaveLength(0);
    expect(store.session).toBeNull();
    expect(store.sessionId).toBe("B");

    const openB2 = store.open("B");
    messagesCalls[1].resolve([msg(7, { session_id: "B", content: "B-retry" })]);
    await openB2;
    expect(store.messages.map((m) => m.id)).toEqual([7]);
  });

  it("stale loadMessages/loadApprovals/loadContext from an older open are dropped", async () => {
    const store = useSessionStore();

    vi.spyOn(api, "messages").mockResolvedValue([msg(1, { session_id: "A" })]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    vi.spyOn(api, "context").mockResolvedValue(contextStats(1));
    vi.spyOn(api, "session").mockResolvedValue(sessionRecord("A"));
    await store.open("A");

    vi.restoreAllMocks();

    const msgD = deferred<Message[]>();
    const apprD = deferred<Approval[]>();
    const ctxD = deferred<ContextStats>();
    const messagesSpy = vi.spyOn(api, "messages").mockReturnValue(msgD.promise);
    const approvalsSpy = vi.spyOn(api, "approvals").mockReturnValue(apprD.promise);
    const contextSpy = vi.spyOn(api, "context").mockReturnValue(ctxD.promise);

    store.applyEvent({
      id: 9,
      type: "message.created",
      session_id: "A",
      payload: { role: "user", message_id: 9 },
      created_at: "2026-01-01T00:00:00Z",
    });
    store.applyEvent({
      id: 10,
      type: "approval.required",
      session_id: "A",
      payload: { status: "awaiting_confirmation" },
      created_at: "2026-01-01T00:00:00Z",
    });
    store.applyEvent({
      id: 11,
      type: "context.compacted",
      session_id: "A",
      payload: { omitted_messages: 2 },
      created_at: "2026-01-01T00:00:00Z",
    });

    expect(messagesSpy).toHaveBeenCalledTimes(2);
    expect(approvalsSpy).toHaveBeenCalled();
    expect(contextSpy).toHaveBeenCalled();

    vi.spyOn(api, "messages").mockResolvedValue([]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "context").mockResolvedValue(contextStats(2));
    vi.spyOn(api, "session").mockResolvedValue(sessionRecord("B"));
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    await store.open("B");

    msgD.resolve([msg(99, { session_id: "A" })]);
    apprD.resolve([approval({ id: "stale-approval" })]);
    ctxD.resolve(contextStats(99));
    await flushPromises();

    expect(store.messages.some((m) => m.id === 99)).toBe(false);
    expect(store.approvals.some((a) => a.id === "stale-approval")).toBe(false);
    expect(store.context?.omitted_messages).not.toBe(99);
  });

  it("a stale send() completion from A must not mutate B", async () => {
    const store = useSessionStore();

    vi.spyOn(api, "messages").mockResolvedValue([]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    vi.spyOn(api, "context").mockResolvedValue(contextStats(0));
    vi.spyOn(api, "session").mockResolvedValue(sessionRecord("A"));
    await store.open("A");
    store.streaming = { content: "in-flight", reasoning: "" };

    const sendD = deferred<Message>();
    const sendSpy = vi.spyOn(api, "sendMessage").mockReturnValue(sendD.promise);
    const sendPromise = store.send("hello");

    vi.spyOn(api, "messages").mockImplementation((id: string) =>
      Promise.resolve(id === "B" ? [msg(5, { session_id: "B", content: "B-msg" })] : []),
    );
    vi.spyOn(api, "session").mockImplementation((id: string) =>
      Promise.resolve(sessionRecord(id)),
    );
    await store.open("B");

    expect(sendSpy).toHaveBeenCalledTimes(1);
    expect(sendSpy).toHaveBeenCalledWith("A", "hello", undefined);

    sendD.resolve(msg(42, { session_id: "A", content: "from-A" }));
    await flushPromises();
    await sendPromise;

    expect(store.messages.some((m) => m.id === 42)).toBe(false);
    expect(store.sessionId).toBe("B");
  });

  it("stale resolve() and stop() results from A do not touch B", async () => {
    const store = useSessionStore();

    vi.spyOn(api, "messages").mockResolvedValue([]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "children").mockResolvedValue([]);
    vi.spyOn(api, "audit").mockResolvedValue([]);
    vi.spyOn(api, "context").mockResolvedValue(contextStats(0));
    vi.spyOn(api, "session").mockResolvedValue(sessionRecord("A"));
    await store.open("A");
    store.session = sessionRecord("A", { status: "pending" });

    const resolveD = deferred<{ session_id: string; decision: string }>();
    const stopD = deferred<{ stopped: boolean }>();
    const resolveSpy = vi.spyOn(api, "resolveApproval").mockReturnValue(resolveD.promise);
    const stopSpy = vi.spyOn(api, "stop").mockReturnValue(stopD.promise);

    const resolvePromise = store.resolve("ap1", "approve");
    const stopPromise = store.stop();

    vi.spyOn(api, "messages").mockImplementation((id: string) =>
      Promise.resolve(id === "B" ? [msg(5, { session_id: "B" })] : []),
    );
    vi.spyOn(api, "session").mockImplementation((id: string) =>
      Promise.resolve(sessionRecord(id)),
    );
    await store.open("B");

    resolveD.resolve({ session_id: "A", decision: "approved" });
    stopD.resolve({ stopped: true });
    await flushPromises();
    await resolvePromise;
    await stopPromise;

    expect(resolveSpy).toHaveBeenCalledTimes(1);
    expect(stopSpy).toHaveBeenCalledTimes(1);
    expect(store.approvals).toHaveLength(0);
    expect(store.session?.status).toBe("completed");
  });
});

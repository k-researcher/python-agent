import { beforeEach, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import { api } from "../api/client";
import type { Session, ServerEvent, Message } from "../api/types";
import { useSessionStore } from "./session";

function stamp<T extends object>(value: T, revision = 1, run_generation = 2): T {
  return Object.assign(value, { revision, run_generation });
}
function session(revision = 1, run_generation = 2, status = "running"): Session {
  return stamp({ id: "s", project_id: "p", parent_id: null, mode: "dev", llm_profile: "main", title: "test", status, error: null, archived: false, created_at: "", updated_at: "" }, revision, run_generation);
}
function frame(type: string, generation = 2, stream_no = 2, stream_id = "new", delta = "new"): ServerEvent {
  return { type, session_id: "s", ephemeral: true, payload: { generation, stream_no, stream_id, seq: 1, delta } } as ServerEvent;
}
function event(type: string, payload: Record<string, unknown>, id = 2): ServerEvent {
  return { type, session_id: "s", id, created_at: "", payload };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
beforeEach(() => {
  vi.restoreAllMocks();
  setActivePinia(createPinia());
  vi.spyOn(api, "session").mockResolvedValue(session());
  vi.spyOn(api, "messages").mockResolvedValue(stamp([]));
  vi.spyOn(api, "approvals").mockResolvedValue(stamp([]));
  vi.spyOn(api, "children").mockResolvedValue(stamp([]));
  vi.spyOn(api, "audit").mockResolvedValue(stamp([]));
  vi.spyOn(api, "context").mockResolvedValue(stamp({ approximate_tokens: 0, omitted_messages: 0, truncated_tool_results: 0, context_window: 8192, reserved_tokens: 1024 }));
});
it("rejects an unknown older stream when its reset was lost", async () => {
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("message.delta"));
  store.applyEvent(frame("message.delta", 2, 1, "never-seen", "late"));
  expect(store.streaming?.content).toBe("new");
  store.applyEvent(frame("stream.reset", 2, 1, "never-seen"));
  expect(store.streaming?.content).toBe("new");
});
it("remembers reset before first delta and preserves the stream maximum", async () => {
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("stream.reset", 2, 3, "closed"));
  store.applyEvent(frame("message.delta", 2, 3, "closed", "late"));
  store.applyEvent(frame("message.delta", 2, 2, "old", "late"));
  expect(store.streaming).toBeNull();
});
it("local Stop closes before the API returns, including unknown stream IDs", async () => {
  const pending = deferred<{ stopped: boolean }>();
  vi.spyOn(api, "stop").mockReturnValue(pending.promise);
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("message.delta"));
  const stopping = store.stop();
  expect(store.streaming).toBeNull();
  store.applyEvent(frame("message.delta", 2, 99, "late"));
  expect(store.streaming).toBeNull();
  pending.resolve({ stopped: true });
  await stopping;
});
it.each(["completed", "stopped", "error", "interrupted"])("terminal %s cannot reopen the same generation", async (status) => {
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(event("session.status", { generation: 2, status }));
  store.applyEvent(event("session.status", { generation: 2, status: "running" }, 3));
  store.applyEvent(frame("message.delta", 2, 99, "unknown"));
  expect(store.streaming).toBeNull();
  expect(store.session?.status).toBe(status);
  store.applyEvent(event("session.status", { generation: 3, status: "running" }, 4));
  store.applyEvent(frame("message.delta", 3, 1, "next", "next"));
  store.applyEvent(event("session.status", { generation: 2, status: "error" }, 5));
  vi.mocked(api.session).mockResolvedValue(session(6, 3));
  vi.mocked(api.messages).mockResolvedValue(stamp([], 6, 3));
  vi.mocked(api.approvals).mockResolvedValue(stamp([], 6, 3));
  vi.mocked(api.children).mockResolvedValue(stamp([], 6, 3));
  vi.mocked(api.audit).mockResolvedValue(stamp([], 6, 3));
  vi.mocked(api.context).mockResolvedValue(stamp({ approximate_tokens: 0, omitted_messages: 0, truncated_tool_results: 0, context_window: 8192, reserved_tokens: 1024 }, 6, 3));
  store.applyEvent(event("message.created", { generation: 2, role: "assistant", stream_no: 99, stream_id: "old" }, 6));
  expect(store.streaming?.content).toBe("next");
  expect(store.session?.status).toBe("running");
  await flushPromises();
  expect(store.streaming?.content).toBe("next");
});
it("repeats dirty reloads until events stop and applies the whole package atomically", async () => {
  const store = useSessionStore();
  await store.open("s");
  const first = deferred<Message[]>();
  const second = deferred<Message[]>();
  const third = deferred<Message[]>();
  vi.mocked(api.messages).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise).mockReturnValueOnce(third.promise);
  const reloading = store.reloadAll();
  store.applyEvent(event("tool.completed", {}, 2));
  first.resolve(stamp([], 1));
  await flushPromises();
  expect(store.session).toEqual(session());
  store.applyEvent(event("approval.resolved", {}, 3));
  second.resolve(stamp([], 2));
  vi.mocked(api.session).mockResolvedValue(session(3));
  vi.mocked(api.approvals).mockResolvedValue(stamp([], 3));
  vi.mocked(api.children).mockResolvedValue(stamp([], 3));
  vi.mocked(api.audit).mockResolvedValue(stamp([], 3));
  vi.mocked(api.context).mockResolvedValue(stamp({ approximate_tokens: 99, omitted_messages: 0, truncated_tool_results: 0, context_window: 8192, reserved_tokens: 1024 }, 3));
  third.resolve(stamp([], 3));
  await reloading;
  expect(store.context?.approximate_tokens).toBe(99);
  expect(store.session).toEqual(session(3));
});

function packageAt(revision: number, run = 2, status = "running"): void {
  vi.mocked(api.session).mockResolvedValue(session(revision, run, status));
  vi.mocked(api.messages).mockResolvedValue(stamp([], revision, run));
  vi.mocked(api.approvals).mockResolvedValue(stamp([], revision, run));
  vi.mocked(api.children).mockResolvedValue(stamp([], revision, run));
  vi.mocked(api.audit).mockResolvedValue(stamp([], revision, run));
  vi.mocked(api.context).mockResolvedValue(stamp({ approximate_tokens: revision, omitted_messages: 0, truncated_tool_results: 0, context_window: 8192, reserved_tokens: 1024 }, revision, run));
}
it("equal stream numbers require the same ID even after reset", async () => {
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("message.delta"));
  store.applyEvent(frame("stream.reset"));
  store.applyEvent(frame("message.delta", 2, 2, "stranger", "late"));
  expect(store.streaming).toBeNull();
  store.applyEvent(frame("message.delta", 2, 3, "next", "next"));
  expect(store.streaming?.content).toBe("next");
});
it("Stop API failure reloads without reopening the closed generation", async () => {
  vi.spyOn(api, "stop").mockRejectedValue(new Error("offline"));
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("message.delta"));
  await store.stop();
  store.applyEvent(frame("message.delta", 2, 77, "late"));
  expect(store.session?.status).toBe("stopped");
  expect(store.streaming).toBeNull();
});
it("reload recovers the durable guard code and rejects late data", async () => {
  packageAt(5, 2, "error");
  vi.mocked(api.session).mockResolvedValue({ ...session(5, 2, "error"), error: "Обнаружен повтор", error_code: "llm_repetition" });
  const store = useSessionStore();
  await store.open("s");
  expect(store.session?.error_code).toBe("llm_repetition");
  store.applyEvent(frame("message.delta", 2, 90, "late"));
  expect(store.streaming).toBeNull();
});
it("mixed revisions never apply part of a package and the next clean pass converges", async () => {
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("message.delta", 2, 5, "live", "live"));
  const pending = deferred<Message[]>();
  vi.mocked(api.messages).mockReturnValueOnce(pending.promise);
  const reload = store.reloadAll();
  packageAt(3);
  pending.resolve(stamp([], 2));
  await reload;
  expect(store.context?.approximate_tokens).toBe(3);
  expect(store.session).toEqual(session(3));
  expect(store.streaming?.content).toBe("live");
  store.applyEvent(frame("message.delta", 2, 4, "old", "late"));
  expect(store.streaming?.content).toBe("live");
});
it("a failed resource preserves the whole previous view and a later reload succeeds", async () => {
  const store = useSessionStore();
  await store.open("s");
  packageAt(2);
  vi.mocked(api.approvals).mockRejectedValueOnce(new Error("offline"));
  await expect(store.reloadAll()).rejects.toThrow("offline");
  expect(store.session).toEqual(session());
  expect(store.context?.approximate_tokens).toBe(0);
  await store.reloadAll();
  expect(store.context?.approximate_tokens).toBe(2);
  expect(store.session).toEqual(session(2));
});

it("reload closes a durably accepted stream and preserves a newer live stream", async () => {
  const store = useSessionStore();
  await store.open("s");
  store.applyEvent(frame("message.delta", 2, 2, "new", "new"));
  packageAt(2);
  vi.mocked(api.messages).mockResolvedValue(Object.assign(stamp([], 2), {
    completed_streams: [{ generation: 2, stream_no: 1, stream_id: "old" }],
  }));
  await store.reloadAll();
  expect(store.streaming?.content).toBe("new");
  packageAt(3);
  vi.mocked(api.messages).mockResolvedValue(Object.assign(stamp([], 3), {
    completed_streams: [{ generation: 2, stream_no: 2, stream_id: "new" }],
  }));
  await store.reloadAll();
  expect(store.streaming).toBeNull();
  store.applyEvent(frame("message.delta", 2, 1, "unknown-old", "late"));
  expect(store.streaming).toBeNull();
});

it("reloads the atomic package after an audit-only revision", async () => {
  const store = useSessionStore();
  await store.open("s");
  packageAt(2);
  store.applyEvent(event("outbound.audit", { audit_id: 1, status: "error", generation: 2 }, 2));
  await flushPromises();
  expect(store.context?.approximate_tokens).toBe(2);
  expect(store.session?.revision).toBe(2);
});

it("does not apply approvals before the matching messages arrive", async () => {
  const store = useSessionStore();
  await store.open("s");
  packageAt(2);
  const pending = deferred<Message[]>();
  vi.mocked(api.messages).mockReturnValueOnce(pending.promise);
  const approvals = stamp([{ id: "approval-2", tool_call_id: "call-2", tool_name: "write_file", arguments: {}, risk_level: "write", status: "pending", comment: null }], 2);
  vi.mocked(api.approvals).mockResolvedValue(approvals);
  const reload = store.reloadAll();
  await flushPromises();
  expect(store.approvals).toEqual(stamp([]));
  expect(store.session?.revision).toBe(1);
  expect(store.context?.approximate_tokens).toBe(0);
  pending.resolve(stamp([], 2));
  await reload;
  expect(store.approvals).toEqual(approvals);
  expect(store.session?.revision).toBe(2);
  expect(store.context?.approximate_tokens).toBe(2);
});

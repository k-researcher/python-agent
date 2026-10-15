import { computed, ref } from "vue";
import { defineStore } from "pinia";
import { api } from "../api/client";
import {
  isStreamDeltaEvent,
  type Approval,
  type ContextStats,
  type Message,
  type OutboundAudit,
  type ServerEvent,
  type Session,
  type StreamDeltaEvent,
  type TimelineItem,
  type ToolCallView,
  type ToolResult,
} from "../api/types";

interface StreamingState {
  content: string;
  reasoning: string;
}

function parseToolResult(content: string | null): ToolResult | null {
  if (typeof content !== "string" || content.trim() === "") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const record = parsed as Record<string, unknown>;
  if (typeof record.success !== "boolean") return null;
  const result: ToolResult = { success: record.success };
  if ("output" in record) result.output = record.output;
  if (typeof record.error === "string") result.error = record.error;
  return result;
}

function parseToolArguments(raw: string): Record<string, unknown> {
  if (raw.trim() === "") return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // fall through
  }
  return { _invalid_arguments: raw };
}

function toolStatus(
  approval: Approval | null,
  result: ToolResult | null,
): ToolCallView["status"] {
  if (approval) {
    if (approval.status === "pending") return "awaiting_approval";
    if (approval.status === "rejected" || approval.status === "cancelled") return "rejected";
    // approved → the call keeps running below
  }
  if (result) return result.success ? "completed" : "error";
  return "pending";
}

function buildToolCalls(
  message: Message,
  resultsByCall: Map<string, ToolResult>,
  approvalsByCall: Map<string, Approval>,
): ToolCallView[] {
  const raw = message.tool_calls ?? [];
  const views: ToolCallView[] = [];
  for (const call of raw) {
    if (typeof call !== "object" || call === null) continue;
    const id = typeof call.id === "string" ? call.id : "";
    const fn = call.function;
    const fnRecord =
      typeof fn === "object" && fn !== null ? (fn as Record<string, unknown>) : {};
    const name = typeof fnRecord.name === "string" && fnRecord.name !== "" ? fnRecord.name : "tool";
    const argumentsRaw = typeof fnRecord.arguments === "string" ? fnRecord.arguments : "";
    const approval = approvalsByCall.get(id) ?? null;
    const result = resultsByCall.get(id) ?? null;
    views.push({
      id,
      name,
      arguments: parseToolArguments(argumentsRaw),
      argumentsRaw,
      result,
      approval,
      status: toolStatus(approval, result),
    });
  }
  return views;
}

export const useSessionStore = defineStore("session", () => {
  const sessionId = ref<string | null>(null);
  const session = ref<Session | null>(null);
  const messages = ref<Message[]>([]);
  const approvals = ref<Approval[]>([]);
  const streaming = ref<StreamingState | null>(null);
  const context = ref<ContextStats | null>(null);
  const audit = ref<OutboundAudit[]>([]);
  const children = ref<Session[]>([]);

  // Delta ordering: one stream_id per generation; seq grows inside a stream per kind.
  // Superseded stream ids are remembered so a straggling fragment from an older
  // generation is dropped instead of being mistaken for a fresh stream.
  let activeStreamId: string | null = null;
  let finishedStreamIds = new Set<string>();
  let contentSeq = 0;
  let reasoningSeq = 0;
  let generation = 0;

  function resetStream(): void {
    streaming.value = null;
    activeStreamId = null;
    finishedStreamIds.clear();
    contentSeq = 0;
    reasoningSeq = 0;
  }

  function clearSessionData(): void {
    messages.value = [];
    approvals.value = [];
    context.value = null;
    audit.value = [];
    children.value = [];
    session.value = null;
    resetStream();
  }

  function close(): void {
    generation += 1;
    sessionId.value = null;
    clearSessionData();
  }

  function isCurrent(id: string | null, gen: number): boolean {
    return id != null && sessionId.value === id && generation === gen;
  }

  async function loadMessages(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    const data = await api.messages(id);
    if (!isCurrent(id, gen)) return;
    messages.value = data;
  }

  async function loadApprovals(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    const data = await api.approvals(id);
    if (!isCurrent(id, gen)) return;
    approvals.value = data;
  }

  async function loadContext(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    const data = await api.context(id);
    if (!isCurrent(id, gen)) return;
    context.value = data;
  }

  async function reloadAll(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    const [msgs, appr, kids, aud, ctx, sess] = await Promise.all([
      api.messages(id),
      api.approvals(id),
      api.children(id),
      api.audit(id),
      api.context(id).catch(() => null),
      api.session(id).catch(() => null),
    ]);
    if (!isCurrent(id, gen)) return;
    messages.value = msgs;
    approvals.value = appr;
    children.value = kids;
    audit.value = aud;
    context.value = ctx;
    if (sess) session.value = sess;
  }

  async function open(id: string): Promise<void> {
    generation += 1;
    sessionId.value = id;
    clearSessionData();
    await reloadAll();
  }

  async function send(content: string, reasoningEffort?: string): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) throw new Error("Нет открытой сессии");
    const created = await api.sendMessage(id, content, reasoningEffort);
    if (!isCurrent(id, gen)) return;
    resetStream();
    // A `message.created` WS event may already have reloaded the list (including the
    // new message) before the POST resolves. Dedup by id so the message is added
    // exactly once (self-check point 4).
    messages.value = [...messages.value.filter((m) => m.id !== created.id), created];
    if (session.value) {
      session.value = { ...session.value, status: "pending", error: null };
    }
  }

  async function resolve(
    approvalId: string,
    decision: "approve" | "reject",
    comment?: string,
  ): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    await api.resolveApproval(approvalId, decision, comment);
    if (!isCurrent(id, gen)) return;
    await Promise.all([loadApprovals(), loadMessages()]);
  }

  async function stop(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    await api.stop(id);
    if (!isCurrent(id, gen)) return;
    if (session.value) {
      session.value = { ...session.value, status: "stopped" };
    }
  }

  function applyDelta(event: StreamDeltaEvent): void {
    const payload = event.payload;
    // Server frames are not validated upstream: a malformed frame must never throw
    // or corrupt a live stream (self-check point 1).
    if (typeof payload !== "object" || payload === null || typeof payload.stream_id !== "string") {
      return;
    }
    const { stream_id, seq, delta } = payload;
    const isMessage = event.type === "message.delta";
    // A frame without actual text never opens or mutates a stream (self-check 1):
    // a broken frame carrying a new stream_id must not wipe the live generation.
    if (typeof delta !== "string") return;
    if (stream_id !== activeStreamId) {
      // A fragment whose stream was already superseded is late: drop it and keep
      // the current generation untouched.
      if (finishedStreamIds.has(stream_id)) return;
      if (activeStreamId != null) finishedStreamIds.add(activeStreamId);
      // A brand-new stream_id starts a fresh generation; drop any queued text.
      activeStreamId = stream_id;
      contentSeq = 0;
      reasoningSeq = 0;
      streaming.value = { content: "", reasoning: "" };
    }
    const current =
      streaming.value ?? { content: "", reasoning: "" };
    if (isMessage) {
      if (seq == null || seq > contentSeq) {
        if (seq != null) contentSeq = seq;
        current.content += delta;
      }
    } else {
      if (seq == null || seq > reasoningSeq) {
        if (seq != null) reasoningSeq = seq;
        current.reasoning += delta;
      }
    }
    streaming.value = current;
  }

  function applyApprovalEvent(): void {
    void loadApprovals();
    void loadMessages();
  }

  /** Fold one server event into the open session state. */
  function applyEvent(event: ServerEvent): void {
    if (isStreamDeltaEvent(event)) {
      if (event.session_id !== sessionId.value) return;
      if (event.type === "stream.reset") {
        resetStream();
        return;
      }
      applyDelta(event);
      return;
    }
    if (event.session_id !== sessionId.value) return;
    switch (event.type) {
      case "message.created":
        resetStream();
        void loadMessages();
        break;
      case "approval.required":
      case "approval.resolved":
      case "approval.cancelled":
        applyApprovalEvent();
        break;
      case "tool.completed":
        void loadMessages();
        break;
      case "context.compacted":
        void loadContext();
        break;
      case "session.status":
      case "session.error":
        if (session.value) {
          const payload = event.payload;
          if (typeof payload !== "object" || payload === null) break;
          const next = { ...session.value };
          if (typeof (payload as Record<string, unknown>).status === "string") {
            next.status = (payload as Record<string, unknown>).status as string;
          }
          if (typeof (payload as Record<string, unknown>).error === "string") {
            next.error = (payload as Record<string, unknown>).error as string;
          }
          session.value = next;
        }
        break;
      case "session.child_created":
        void loadChildren();
        break;
      default:
        break;
    }
  }

  async function loadChildren(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    const data = await api.children(id);
    if (!isCurrent(id, gen)) return;
    children.value = data;
  }

  /** Merged feed: messages (minus skipped), tool calls with results and approvals, summary, streaming. */
  const timeline = computed<TimelineItem[]>(() => {
    const resultsByCall = new Map<string, ToolResult>();
    for (const m of messages.value) {
      if (m.skipped || m.role !== "tool" || m.tool_call_id == null) continue;
      const result = parseToolResult(m.content);
      if (result) resultsByCall.set(m.tool_call_id, result);
    }
    const approvalsByCall = new Map<string, Approval>();
    for (const a of approvals.value) {
      approvalsByCall.set(a.tool_call_id, a);
    }

    const items: TimelineItem[] = [];
    for (const m of messages.value) {
      if (m.skipped) continue;
      if (m.role === "user") {
        items.push(
          m.kind === "summary" ? { kind: "summary", message: m } : { kind: "user", message: m },
        );
        continue;
      }
      if (m.role === "assistant") {
        const calls = buildToolCalls(m, resultsByCall, approvalsByCall);
        const hasText = m.content != null && m.content.trim() !== "";
        if (hasText || calls.length === 0) {
          items.push({ kind: "assistant", message: m });
        }
        for (const call of calls) {
          items.push({ kind: "tool", call });
        }
      }
      // role === "tool" messages are folded into their assistant card.
    }
    const stream = streaming.value;
    if (stream && (stream.content !== "" || stream.reasoning !== "")) {
      items.push({ kind: "streaming", content: stream.content, reasoning: stream.reasoning });
    }
    return items;
  });

  return {
    sessionId,
    session,
    messages,
    approvals,
    streaming,
    context,
    audit,
    children,
    close,
    open,
    send,
    resolve,
    stop,
    applyEvent,
    timeline,
  };
});
